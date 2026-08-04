//! Lowering of the cubecl structured `scf` dialect to an unstructured LLVM CFG.

use cubecl_core::ir::dialect::BlockPtrExt;
use cubecl_core::ir::dialect::branch::{self, ConditionOp, IsExitTerminator};
use cubecl_core::ir::dialect::cmp::{SLessThanOp, ULessThanOp};
use cubecl_core::ir::dialect::general::CastOp;
use cubecl_core::ir::dialect::math::IAddOp;
use cubecl_core::ir::dialect::scf::{IfOp, RangeLoopOp, SwitchOp, WhileOp};
use cubecl_core::ir::interfaces::ScalarType;
use cubecl_core::ir::prelude::*;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::irbuild::inserter::{BlockInsertionPoint, OpInsertionPoint};
use pliron::region::Region;
use pliron_llvm::ops as llvm;

#[op_interface]
pub trait LowerCpuCF {
    verify_op_succ!();
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}

#[op_interface_impl]
impl LowerCpuCF for IfOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let then_block = self.then_block(ctx);
        let else_block = self.else_block(ctx);
        let then_region = self.then_region(ctx);
        let else_region = self.else_region(ctx);
        let then_term = terminator(ctx, then_block);
        let else_term = terminator(ctx, else_block);

        // Split off the merge block so ops after the `if` continue there.
        let (pre, merge) = split_join_block(ctx, rewriter, op, "if_merge");

        rewriter.set_insertion_point_to_block_end(pre);
        let cond = self.condition(ctx);
        let cond_br = llvm::CondBrOp::new(ctx, cond, then_block, vec![], else_block, vec![]);
        rewriter.append_op(ctx, &cond_br);

        branch_to_yielded(ctx, rewriter, then_term, merge);
        branch_to_yielded(ctx, rewriter, else_term, merge);

        rewriter.inline_region(ctx, then_region, BlockInsertionPoint::AfterBlock(pre));
        rewriter.inline_region(
            ctx,
            else_region,
            BlockInsertionPoint::AfterBlock(then_block),
        );

        let results = merge.arguments(ctx);
        rewriter.replace_operation_with_values(ctx, op, results);
        Ok(())
    }
}

#[op_interface_impl]
impl LowerCpuCF for WhileOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let init = self.initial_carried_values(ctx);
        let before_block = self.before_block(ctx);
        let before_region = self.before_region(ctx);
        let after_block = self.after_block(ctx);
        let after_region = self.after_region(ctx);
        let before_term = terminator(ctx, before_block);
        let after_term = terminator(ctx, after_block);

        let (pre, exit) = split_join_block(ctx, rewriter, op, "while_exit");

        rewriter.set_insertion_point_to_block_end(pre);
        let br_to_before = llvm::BrOp::new(ctx, before_block, init);
        rewriter.append_op(ctx, &br_to_before);

        // The `before` region decides whether to run another iteration. The values it forwards go
        // to the `after` block when the loop continues, and out of the loop when it doesn't.
        if let Some(condition) = before_term.as_op::<ConditionOp>(ctx) {
            let cond = condition.condition(ctx);
            let forwarded = condition.forward_values(ctx);
            rewriter.set_insertion_point_before_operation(before_term);
            let cond_br =
                llvm::CondBrOp::new(ctx, cond, after_block, forwarded.clone(), exit, forwarded);
            rewriter.append_op(ctx, &cond_br);
            rewriter.erase_operation(ctx, before_term);
        } else {
            assert!(
                before_term.impls::<dyn IsExitTerminator>(ctx),
                "`while` condition must be terminated with `branch.condition`"
            );
        }

        // Back-edge to the condition.
        branch_to_yielded(ctx, rewriter, after_term, before_block);

        rewriter.inline_region(ctx, before_region, BlockInsertionPoint::AfterBlock(pre));
        rewriter.inline_region(
            ctx,
            after_region,
            BlockInsertionPoint::AfterBlock(before_block),
        );

        let results = exit.arguments(ctx);
        rewriter.replace_operation_with_values(ctx, op, results);
        Ok(())
    }
}

#[op_interface_impl]
impl LowerCpuCF for RangeLoopOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let start = self.start(ctx);
        let end = self.end(ctx);
        let step = self.step(ctx);
        let init = self.initial_carried_values(ctx);
        let body_block = self.loop_body(ctx);
        let body_region = self.loop_region(ctx);
        let body_term = terminator(ctx, body_block);

        let signed = type_cast::<dyn ScalarType>(&*end.get_type(ctx).deref(ctx))
            .map(|ty| ty.elem_type(ctx).is_signed_int())
            .unwrap_or(false);

        let (pre, exit) = split_join_block(ctx, rewriter, op, "for_exit");

        // The header carries the induction variable followed by the loop carried values, in the
        // same order as the body block arguments it feeds.
        let header_args = body_block
            .arguments(ctx)
            .into_iter()
            .map(|arg| arg.get_type(ctx))
            .collect();
        let header = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AfterBlock(pre),
            Some("for_header".try_into().unwrap()),
            header_args,
        );

        rewriter.set_insertion_point_to_block_end(pre);
        let mut entry_args = vec![start];
        entry_args.extend(init);
        let br_to_header = llvm::BrOp::new(ctx, header, entry_args);
        rewriter.append_op(ctx, &br_to_header);

        rewriter.set_insertion_point_to_block_end(header);
        let header_args = header.arguments(ctx);
        let cond = less_than(ctx, rewriter, signed, header_args[0], end);
        let cond_br = llvm::CondBrOp::new(
            ctx,
            cond,
            body_block,
            header_args.clone(),
            exit,
            header_args[1..].to_vec(),
        );
        rewriter.append_op(ctx, &cond_br);

        // The back-edge steps the induction variable and forwards the values yielded by the body.
        if !body_term.impls::<dyn IsExitTerminator>(ctx) {
            rewriter.set_insertion_point_before_operation(body_term);
            let next = IAddOp::new(ctx, self.iter_var(ctx), step);
            rewriter.append_op(ctx, &next);
            let mut back_args = vec![next.get_result(ctx)];
            back_args.extend(body_term.operands(ctx));
            let back_edge = llvm::BrOp::new(ctx, header, back_args);
            rewriter.append_op(ctx, &back_edge);
            rewriter.erase_operation(ctx, body_term);
        }

        rewriter.inline_region(ctx, body_region, BlockInsertionPoint::AfterBlock(header));

        let results = exit.arguments(ctx);
        rewriter.replace_operation_with_values(ctx, op, results);
        Ok(())
    }
}

#[op_interface_impl]
impl LowerCpuCF for SwitchOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let value = self.value(ctx);
        let default_block = self.default_block(ctx);
        let default_region = self.default_region(ctx);
        let cases = self.cases(ctx);

        let elem = type_cast::<dyn ScalarType>(&*value.get_type(ctx).deref(ctx))
            .expect("switch value must be a scalar type")
            .elem_type(ctx);
        let int_ty = elem.to_type(ctx);

        let (pre, merge) = split_join_block(ctx, rewriter, op, "switch_merge");

        let case_regions: Vec<Ptr<Region>> = (0..cases.len())
            .map(|i| op.deref(ctx).get_region(i + 1))
            .collect();

        let default_term = terminator(ctx, default_block);
        branch_to_yielded(ctx, rewriter, default_term, merge);

        let mut switch_cases = Vec::with_capacity(cases.len());
        for (const_val, block) in &cases {
            let term = terminator(ctx, *block);
            branch_to_yielded(ctx, rewriter, term, merge);
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
            let value = IntegerAttr::new(i32_ty, const_val.value());
            switch_cases.push(llvm::SwitchCase {
                value,
                dest: *block,
                dest_opds: vec![],
            });
        }

        rewriter.set_insertion_point_to_block_end(pre);
        let cond_int = CastOp::new(ctx, int_ty, value);
        rewriter.append_op(ctx, &cond_int);
        let switch = llvm::SwitchOp::new(
            ctx,
            cond_int.get_result(ctx),
            default_block,
            vec![],
            switch_cases,
        );
        rewriter.append_op(ctx, &switch);

        rewriter.inline_region(ctx, default_region, BlockInsertionPoint::AfterBlock(pre));
        let mut prev = default_block;
        for (region, (_, block)) in case_regions.into_iter().zip(cases.iter()) {
            rewriter.inline_region(ctx, region, BlockInsertionPoint::AfterBlock(prev));
            prev = *block;
        }

        let results = merge.arguments(ctx);
        rewriter.replace_operation_with_values(ctx, op, results);
        Ok(())
    }
}

#[op_interface_impl]
impl LowerCpuCF for branch::ReturnOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.get_operation().operands(ctx).into_iter().next();
        let ret = llvm::ReturnOp::new(ctx, value);
        rewriter.append_op(ctx, &ret);
        rewriter.replace_operation(ctx, self.get_operation(), ret.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl LowerCpuCF for branch::UnreachableOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let unreachable = llvm::UnreachableOp::new(ctx);
        rewriter.append_op(ctx, &unreachable);
        rewriter.replace_operation(ctx, self.get_operation(), unreachable.get_operation());
        Ok(())
    }
}

fn terminator(ctx: &Context, block: Ptr<BasicBlock>) -> Ptr<Operation> {
    block
        .deref(ctx)
        .get_terminator(ctx)
        .expect("structured region blocks must be terminated")
}

/// Split the block holding `op` so that everything following it becomes the block the lowered
/// control flow joins back into. The join block takes the results of `op` as block arguments.
fn split_join_block(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    label: &str,
) -> (Ptr<BasicBlock>, Ptr<BasicBlock>) {
    let pre = op
        .deref(ctx)
        .get_parent_block()
        .expect("structured op must be in a block");
    let join = rewriter.split_block(
        ctx,
        pre,
        OpInsertionPoint::BeforeOperation(op),
        Some(label.try_into().unwrap()),
    );
    for ty in op.result_types(ctx) {
        BasicBlock::push_argument(join, ctx, ty);
    }
    (pre, join)
}

/// Replace a `branch.yield` terminator with a branch to `dest`, forwarding the yielded values as
/// block arguments. Terminators that leave the function are kept as is, they lower on their own.
fn branch_to_yielded(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    terminator: Ptr<Operation>,
    dest: Ptr<BasicBlock>,
) {
    if terminator.impls::<dyn IsExitTerminator>(ctx) {
        return;
    }
    let args = terminator.operands(ctx);
    rewriter.set_insertion_point_before_operation(terminator);
    let br = llvm::BrOp::new(ctx, dest, args);
    rewriter.append_op(ctx, &br);
    rewriter.erase_operation(ctx, terminator);
}

fn less_than(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    signed: bool,
    lhs: Value,
    rhs: Value,
) -> Value {
    if signed {
        let op = SLessThanOp::new(ctx, lhs, rhs);
        rewriter.append_op(ctx, &op);
        op.get_result(ctx)
    } else {
        let op = ULessThanOp::new(ctx, lhs, rhs);
        rewriter.append_op(ctx, &op);
        op.get_result(ctx)
    }
}

pub type SCFToLlvmCf = DialectConversionPass<CfToLlvmConversion>;

#[derive(Default)]
pub struct CfToLlvmConversion;

impl DialectConversion for CfToLlvmConversion {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn LowerCpuCF>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        op_cast::<dyn LowerCpuCF>(&*op.dyn_op(ctx))
            .unwrap()
            .rewrite(ctx, rewriter, operands_info)
    }
}
