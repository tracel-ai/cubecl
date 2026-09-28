//! Structured control flow lowering.

use crate::prelude::*;
#[cfg(feature = "nvptx")]
use crate::shared::loop_hints::LoopHint;
#[cfg(feature = "nvptx")]
use cubecl_core::ir::dialect::memory::{DeclareVariableOp, IndexOp};
use cubecl_core::ir::dialect::{
    BlockPtrExt,
    branch::{self, ConditionOp, IsExitTerminator},
    cmp::{SLessThanOp, ULessThanOp},
    general::CastOp,
    math::IAddOp,
    scf::{IfOp, RangeLoopOp, SwitchOp, WhileOp},
};
use pliron::region::Region;

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

        #[cfg(feature = "nvptx")]
        let shape = LoopShape::new(ctx, op, start, end, step);

        let signed = type_cast::<dyn ScalarType>(&*end.get_type(ctx).deref(ctx))
            .map(|ty| ty.elem_type(ctx).is_signed_int())
            .unwrap_or(false);

        let (pre, exit) = split_join_block(ctx, rewriter, op, "for_exit");

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

        if !body_term.impls::<dyn IsExitTerminator>(ctx) {
            rewriter.set_insertion_point_before_operation(body_term);
            let next = IAddOp::new(ctx, self.iter_var(ctx), step);
            rewriter.append_op(ctx, &next);
            let mut back_args = vec![next.get_result(ctx)];
            back_args.extend(body_term.operands(ctx));
            let back_edge = llvm::BrOp::new(ctx, header, back_args);
            rewriter.append_op(ctx, &back_edge);
            #[cfg(feature = "nvptx")]
            shape.mark(ctx, back_edge.get_operation());
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

/// What a loop's structure says about how it should be unrolled, which LLVM's cost model
/// cannot see once the loop is lowered. Only NVPTX asks for a loop hint, so only a build with
/// that target reads the shape.
#[cfg(feature = "nvptx")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoopShape {
    /// A trip count known at compile time, and a body indexing a local array.
    ConstantOverLocalArray,
    Other,
}

#[cfg(feature = "nvptx")]
impl LoopShape {
    /// What this loop's target should be asked for, or `None` to leave it to LLVM's cost model.
    ///
    /// A loop of a constant trip count over a local array is worth unrolling completely on
    /// NVPTX, as NVVM does: only then is every index a constant, and a local array indexed by
    /// constants alone becomes registers instead of local memory. LLVM's NVPTX cost model stops
    /// short of that for a loop of more than a handful of steps (a top-k accumulator's 64, say),
    /// and the array it leaves in local memory costs several times the loop. AMDGPU's cost model
    /// already raises its threshold for a loop that touches a private array, so it needs no hint.
    fn hint(self, target: LlvmTarget) -> Option<LoopHint> {
        match self {
            LoopShape::ConstantOverLocalArray => match target {
                LlvmTarget::Nvptx => Some(LoopHint::UnrollFull),
                _ => None,
            },
            LoopShape::Other => None,
        }
    }

    /// Gives the loop `latch` closes the hint its target wants.
    fn mark(self, ctx: &Context, latch: Ptr<Operation>) {
        if let Some(hint) = self.hint(ctx.target()) {
            hint.attach(ctx, latch);
        }
    }

    /// The bounds are checked first: they are read off their definitions, where the array
    /// check walks the body.
    fn new(ctx: &Context, op: Ptr<Operation>, start: Value, end: Value, step: Value) -> Self {
        let constant = |value: Value| {
            value
                .defining_op()
                .is_some_and(|def| Operation::get_op::<ConstantOp>(def, ctx).is_some())
        };
        if constant(start) && constant(end) && constant(step) && indexes_local_array(ctx, op) {
            LoopShape::ConstantOverLocalArray
        } else {
            LoopShape::Other
        }
    }
}

/// Whether the body of `op` indexes a local array, stopping at the first index that does.
#[cfg(feature = "nvptx")]
fn indexes_local_array(ctx: &Context, op: Ptr<Operation>) -> bool {
    use pliron::graph::walkers::{
        IRNode, WALKCONFIG_PREORDER_FORWARD,
        interruptible::{immutable::walk_op, walk_advance, walk_break},
    };

    walk_op(
        ctx,
        &mut (),
        &WALKCONFIG_PREORDER_FORWARD,
        op,
        |ctx, _, node| {
            let IRNode::Operation(op) = node else {
                return walk_advance();
            };
            let local = op.as_op::<IndexOp>(ctx).is_some_and(|index| {
                index
                    .base(ctx)
                    .defining_op()
                    .and_then(|def| Operation::get_op::<DeclareVariableOp>(def, ctx))
                    .is_some_and(|declare| declare.addr_space(ctx).0 == AddressSpace::Local)
            });
            if local {
                walk_break(())
            } else {
                walk_advance()
            }
        },
    )
    .is_break()
}

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

#[derive(Default, NamedRewrite)]
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

#[cfg(all(test, feature = "nvptx"))]
mod tests {
    use super::*;

    #[test]
    fn only_nvptx_is_asked_to_unroll_a_constant_loop_over_a_local_array() {
        let asked = |target| LoopShape::ConstantOverLocalArray.hint(target);
        assert_eq!(asked(LlvmTarget::Nvptx), Some(LoopHint::UnrollFull));
        // AMDGPU's own cost model already unrolls these, and the CPU has no such loop hint.
        #[cfg(feature = "amdgpu")]
        assert_eq!(asked(LlvmTarget::AmdGpu), None);
        assert_eq!(asked(LlvmTarget::Cpu), None);
    }

    #[test]
    fn any_other_loop_is_left_to_the_cost_model() {
        for target in [
            LlvmTarget::Nvptx,
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu,
            LlvmTarget::Cpu,
        ] {
            assert_eq!(LoopShape::Other.hint(target), None, "{target:?}");
        }
    }
}
