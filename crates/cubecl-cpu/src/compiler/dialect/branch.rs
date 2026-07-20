//! Lowering of the cubecl structured `branch` dialect to an unstructured LLVM CFG.

use cubecl_core::ir::dialect::branch::{self, IfOp, RangeLoopOp, SwitchOp, WhileOp};
use cubecl_core::ir::dialect::cmp::{SLessThanOp, ULessThanOp};
use cubecl_core::ir::dialect::general::CastOp;
use cubecl_core::ir::dialect::math::IAddOp;
use cubecl_core::ir::dialect::memory::{LoadOp, StoreOp};
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

/// Return `cond` if it is already an `i1`, otherwise materialize one with a `cube.cast`
/// inserted at the current insertion point.
fn ensure_i1(ctx: &mut Context, rewriter: &mut DialectConversionRewriter, cond: Value) -> Value {
    let cond_ty = cond.get_type(ctx);
    if let Some(int_ty) = cond_ty.deref(ctx).downcast_ref::<IntegerType>()
        && int_ty.width() == 1
        && int_ty.signedness() == Signedness::Signless
    {
        return cond;
    }
    let i1 = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let cast = CastOp::new(ctx, i1, cond);
    rewriter.append_op(ctx, &cast);
    cast.get_result(ctx)
}

#[op_interface_impl]
impl LowerCpuCF for IfOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let then_block = self.then_block(ctx);
        let else_block = self.else_block(ctx);
        let then_region = self.then_region(ctx);
        let else_region = self.else_region(ctx);
        let then_term = then_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("then block must be terminated");
        let else_term = else_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("else block must be terminated");

        // Split off the merge block so ops after the `if` continue there.
        let pre = self
            .get_operation()
            .deref(ctx)
            .get_parent_block()
            .expect("IfOp must be in a block");
        let merge = rewriter.split_block(
            ctx,
            pre,
            OpInsertionPoint::BeforeOperation(self.get_operation()),
            Some("if_merge".try_into().unwrap()),
        );

        rewriter.set_insertion_point_to_block_end(pre);
        let cond = ensure_i1(ctx, rewriter, self.condition(ctx));
        let cond_br = llvm::CondBrOp::new(ctx, cond, then_block, vec![], else_block, vec![]);
        rewriter.append_op(ctx, &cond_br);

        replace_terminator_with_branch(ctx, rewriter, then_term, merge);
        replace_terminator_with_branch(ctx, rewriter, else_term, merge);

        rewriter.inline_region(ctx, then_region, BlockInsertionPoint::AfterBlock(pre));
        rewriter.inline_region(
            ctx,
            else_region,
            BlockInsertionPoint::AfterBlock(then_block),
        );

        rewriter.erase_operation(ctx, self.get_operation());
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
        let cond_ptr = self.cond_ptr(ctx);
        let body_block = self.loop_body(ctx);
        let body_region = self.get_operation().deref(ctx).get_region(0);
        let body_term = body_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("while body must be terminated");

        let pre = self
            .get_operation()
            .deref(ctx)
            .get_parent_block()
            .expect("WhileOp must be in a block");
        let exit = rewriter.split_block(
            ctx,
            pre,
            OpInsertionPoint::BeforeOperation(self.get_operation()),
            Some("while_exit".try_into().unwrap()),
        );
        let header = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AfterBlock(pre),
            Some("while_header".try_into().unwrap()),
            vec![],
        );

        rewriter.set_insertion_point_to_block_end(pre);
        let br_to_header = llvm::BrOp::new(ctx, header, vec![]);
        rewriter.append_op(ctx, &br_to_header);

        rewriter.set_insertion_point_to_block_end(header);
        let load = LoadOp::new(ctx, cond_ptr);
        rewriter.append_op(ctx, &load);
        let cond = ensure_i1(ctx, rewriter, load.get_result(ctx));
        let cond_br = llvm::CondBrOp::new(ctx, cond, body_block, vec![], exit, vec![]);
        rewriter.append_op(ctx, &cond_br);

        // Body back-edge to the header.
        replace_terminator_with_branch(ctx, rewriter, body_term, header);
        rewriter.inline_region(ctx, body_region, BlockInsertionPoint::AfterBlock(header));

        rewriter.erase_operation(ctx, self.get_operation());
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
        let iter_var = self.iter_var(ctx);
        let start = self.start(ctx);
        let end = self.end(ctx);
        let step = self.step(ctx);
        let body_block = self.loop_body(ctx);
        let body_region = self.loop_region(ctx);
        let body_term = body_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("range_loop body must be terminated");

        let signed = type_cast::<dyn ScalarType>(&*end.get_type(ctx).deref(ctx))
            .map(|ty| ty.elem_type(ctx).is_signed_int())
            .unwrap_or(false);

        let pre = self
            .get_operation()
            .deref(ctx)
            .get_parent_block()
            .expect("RangeLoopOp must be in a block");
        let exit = rewriter.split_block(
            ctx,
            pre,
            OpInsertionPoint::BeforeOperation(self.get_operation()),
            Some("for_exit".try_into().unwrap()),
        );
        let header = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AfterBlock(pre),
            Some("for_header".try_into().unwrap()),
            vec![],
        );

        rewriter.set_insertion_point_to_block_end(pre);
        let store_start = StoreOp::new(ctx, iter_var, start);
        rewriter.append_op(ctx, &store_start);
        let br_to_header = llvm::BrOp::new(ctx, header, vec![]);
        rewriter.append_op(ctx, &br_to_header);

        rewriter.set_insertion_point_to_block_end(header);
        let load = LoadOp::new(ctx, iter_var);
        rewriter.append_op(ctx, &load);
        let cmp = less_than(ctx, rewriter, signed, load.get_result(ctx), end);
        let cond = ensure_i1(ctx, rewriter, cmp);
        let cond_br = llvm::CondBrOp::new(ctx, cond, body_block, vec![], exit, vec![]);
        rewriter.append_op(ctx, &cond_br);

        rewriter.inline_region(ctx, body_region, BlockInsertionPoint::AfterBlock(header));
        let latch = rewriter.create_block(
            ctx,
            BlockInsertionPoint::AfterBlock(body_block),
            Some("for_latch".try_into().unwrap()),
            vec![],
        );
        replace_terminator_with_branch(ctx, rewriter, body_term, latch);

        rewriter.set_insertion_point_to_block_end(latch);
        let load_latch = LoadOp::new(ctx, iter_var);
        rewriter.append_op(ctx, &load_latch);
        let add = IAddOp::new(ctx, load_latch.get_result(ctx), step);
        rewriter.append_op(ctx, &add);
        let store_step = StoreOp::new(ctx, iter_var, add.get_result(ctx));
        rewriter.append_op(ctx, &store_step);
        let back_edge = llvm::BrOp::new(ctx, header, vec![]);
        rewriter.append_op(ctx, &back_edge);

        rewriter.erase_operation(ctx, self.get_operation());
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
        let value = self.value(ctx);
        let default_block = self.default_block(ctx);
        let default_region = self.default_region(ctx);
        let cases = self.cases(ctx);

        let elem = type_cast::<dyn ScalarType>(&*value.get_type(ctx).deref(ctx))
            .expect("switch value must be a scalar type")
            .elem_type(ctx);
        let int_ty = elem.to_type(ctx);

        let pre = self
            .get_operation()
            .deref(ctx)
            .get_parent_block()
            .expect("SwitchOp must be in a block");
        let merge = rewriter.split_block(
            ctx,
            pre,
            OpInsertionPoint::BeforeOperation(self.get_operation()),
            Some("switch_merge".try_into().unwrap()),
        );

        let case_regions: Vec<Ptr<Region>> = (0..cases.len())
            .map(|i| self.get_operation().deref(ctx).get_region(i + 1))
            .collect();

        let default_term = default_block
            .deref(ctx)
            .get_terminator(ctx)
            .expect("switch default must be terminated");
        replace_terminator_with_branch(ctx, rewriter, default_term, merge);

        let mut switch_cases = Vec::with_capacity(cases.len());
        for (const_val, block) in &cases {
            let term = block
                .deref(ctx)
                .get_terminator(ctx)
                .expect("switch case must be terminated");
            replace_terminator_with_branch(ctx, rewriter, term, merge);
            let attr = const_val
                .as_attribute(ctx, elem)
                .downcast::<IntegerAttr>()
                .expect("switch case value must be an integer");
            switch_cases.push(llvm::SwitchCase {
                value: *attr,
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

        rewriter.erase_operation(ctx, self.get_operation());
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
        let ret = llvm::ReturnOp::new(ctx, self.value(ctx));
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

fn replace_terminator_with_branch(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    terminator: Ptr<Operation>,
    dest: Ptr<BasicBlock>,
) {
    rewriter.set_insertion_point_before_operation(terminator);
    let br = llvm::BrOp::new(ctx, dest, vec![]);
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

pub type CfToLlvmConversionPass = DialectConversionPass<CfToLlvmConversion>;

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
