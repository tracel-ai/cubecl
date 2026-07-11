use cubecl_core::ir::{
    NoMemoryEffect,
    attributes::BoolAttr,
    dialect::branch::{IfOp, YieldOp},
    prelude::*,
    types::scalar::BoolType,
};
use cubecl_ir::dialect::branch;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    irbuild::inserter::{BlockInsertionPoint, Inserter},
    opts::constants::BranchOpFoldInterface,
};
use pliron_spirv::ops::{self, BranchOp, MergeOp, SelectionOp};

use crate::ops::to_spirv_dialect::ToSpirvDialectOp;

// Custom branch because of `BoolType`
#[pliron_op(
    name = "spirv.cube.branch_conditional",
    format = "succ($0) `(` operands(CharSpace(`,`)) `)`",
    operands = (condition: BoolType, true_dest_opds, false_dest_opds),
    verifier = "succ"
)]
#[op_interfaces(IsTerminatorInterface, NResultsInterface<0>, NSuccsInterface<2>, OperandSegmentInterface)]
#[op_traits(NoMemoryEffect)]
pub struct BranchConditionalOp;

impl BranchConditionalOp {
    /// Create a new [`BranchConditionalOp`].
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        true_dest: Ptr<BasicBlock>,
        true_dest_opds: Vec<Value>,
        false_dest: Ptr<BasicBlock>,
        false_dest_opds: Vec<Value>,
    ) -> Self {
        let (operands, segment_sizes) =
            Self::compute_segment_sizes(vec![vec![condition], true_dest_opds, false_dest_opds]);

        let op = BranchConditionalOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![true_dest, false_dest],
                0,
            ),
        };

        // Set the operand segment sizes attribute.
        op.set_operand_segment_sizes(ctx, segment_sizes);
        op
    }
}

#[op_interface_impl]
impl BranchOpInterface for BranchConditionalOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        // Skip the first segment, which is the condition.
        self.get_segment(ctx, succ_idx + 1)
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        // The successor operands start at segment 1, since segment 0 is the condition operand.
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        // The successor operands start at segment 1, since segment 0 is the condition operand.
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }
}

impl BranchConditionalOp {
    fn possible_successor_indices(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<usize> {
        let Some(cond_attr) = operands.first().unwrap().as_ref() else {
            let num_successors = self.get_operation().deref(ctx).successors().count();
            return (0..num_successors).collect();
        };
        let cond = cond_attr
            .downcast_ref::<BoolAttr>()
            .expect("CondBrOp condition operand must be an IntegerAttr");
        let taken = if cond.0 { 0 } else { 1 };
        vec![taken]
    }
}

#[op_interface_impl]
impl BranchOpFoldInterface for BranchConditionalOp {
    fn check_fold(&self, ctx: &Context, operands: &[Option<AttrObj>]) -> Vec<Ptr<BasicBlock>> {
        let successors: Vec<Ptr<BasicBlock>> =
            self.get_operation().deref(ctx).successors().collect();

        self.possible_successor_indices(ctx, operands)
            .iter()
            .map(|ind| successors[*ind])
            .collect()
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        ops: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        let possible_successor_indices = self.possible_successor_indices(ctx, ops);
        if possible_successor_indices.len() != 1 {
            return IRStatus::Unchanged;
        };
        let successor_ind = possible_successor_indices[0];
        let successors: Vec<Ptr<BasicBlock>> =
            self.get_operation().deref(ctx).successors().collect();
        let new_op = BranchOp::new(
            ctx,
            successors[successor_ind],
            self.successor_operands(ctx, successor_ind),
        )
        .get_operation();
        let old_op = self.get_operation();
        rewriter.insert_operation(ctx, new_op);
        rewriter.replace_operation(ctx, old_op, new_op);
        IRStatus::Changed
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for BranchConditionalOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let cond = self.get_operand_condition(ctx);
        let true_dest = self.get_operation().deref(ctx).get_successor(0);
        let true_opds = self.successor_operands(ctx, 0);
        let false_dest = self.get_operation().deref(ctx).get_successor(1);
        let false_opds = self.successor_operands(ctx, 1);
        let op =
            ops::BranchConditionalOp::new(ctx, cond, true_dest, true_opds, false_dest, false_opds);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation(ctx, self.get_operation(), op.get_operation());
        Ok(())
    }
}

#[op_interface]
pub trait ToSpirvCFDialect {
    verify_op_succ!();
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}

#[op_interface_impl]
impl ToSpirvCFDialect for IfOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let selection = SelectionOp::new(ctx, vec![]);
        let select_region = selection.region(ctx);
        let entry = selection.entry_block(ctx);
        rewriter.set_insertion_point_to_block_end(entry);

        let then_region = self.then_region(ctx);
        let then_block = self.then_block(ctx);
        let then_term = then_block.deref(ctx).get_terminator(ctx).unwrap();
        let else_region = self.else_region(ctx);
        let else_block = self.else_block(ctx);
        let else_term = else_block.deref(ctx).get_terminator(ctx).unwrap();

        let merge = BasicBlock::new(ctx, None, vec![]);

        assert!(then_term.is_op::<YieldOp>(ctx), "TODO");
        assert!(else_term.is_op::<YieldOp>(ctx), "TODO");

        let branch_cond = BranchConditionalOp::new(
            ctx,
            self.condition(ctx),
            then_block,
            vec![],
            else_block,
            vec![],
        );
        rewriter.insert_op(ctx, &branch_cond);

        rewriter.set_insertion_point_before_operation(then_term);
        let then_branch = BranchOp::new(ctx, merge, vec![]);
        rewriter.insert_op(ctx, &then_branch);
        rewriter.erase_operation(ctx, then_term);
        rewriter.inline_region(ctx, then_region, BlockInsertionPoint::AfterBlock(entry));

        rewriter.set_insertion_point_before_operation(else_term);
        let else_branch = BranchOp::new(ctx, merge, vec![]);
        rewriter.insert_op(ctx, &else_branch);
        rewriter.erase_operation(ctx, else_term);
        rewriter.inline_region(
            ctx,
            else_region,
            BlockInsertionPoint::AfterBlock(then_block),
        );

        rewriter.insert_block(ctx, BlockInsertionPoint::AtRegionEnd(select_region), merge);
        rewriter.set_insertion_point_to_block_end(merge);

        let merge_op = MergeOp::new(ctx, vec![]);
        rewriter.insert_op(ctx, &merge_op);

        rewriter.set_insertion_point_before_operation(self.get_operation());
        rewriter.insert_op(ctx, &selection);
        rewriter.replace_operation(ctx, self.get_operation(), selection.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvCFDialect for branch::ReturnOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let opds = op.operands(ctx);
        let return_ = ops::ReturnOp::new(ctx, opds.into_iter().next());
        rewriter.insert_op(ctx, &return_);
        rewriter.replace_operation(ctx, op, return_.get_operation());

        Ok(())
    }
}

pub type BranchToSpirvConversionPass = DialectConversionPass<BranchToSpirvConversion>;

#[derive(Default)]
pub struct BranchToSpirvConversion;

impl DialectConversion for BranchToSpirvConversion {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn ToSpirvCFDialect>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        op_cast::<dyn ToSpirvCFDialect>(&*op.dyn_op(ctx))
            .unwrap()
            .rewrite(ctx, rewriter, operands_info)
    }
}
