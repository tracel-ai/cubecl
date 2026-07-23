use cubecl_core::{
    define_scalar,
    frontend::{AddNativeExpand, PartialOrdNativeExpand},
    ir::{
        NoMemoryEffect,
        dialect::branch::{IfOp, YieldOp},
        prelude::*,
        types::scalar::BoolType,
    },
};
use cubecl_ir::{
    Scope,
    dialect::{
        branch::{self, IsExitTerminator, RangeLoopOp, SwitchOp, WhileOp},
        memory::{LoadOp, StoreOp},
    },
};
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    irbuild::inserter::{BlockInsertionPoint, Inserter},
    opts::constants::BranchOpFoldInterface,
};
use pliron_spirv::ops::{self, BranchOp, LoopOp, MergeOp, SelectionOp};

use crate::ops::to_spirv_dialect::ToSpirvDialectOp;

// Custom branch because of `BoolType`
#[pliron_op(
    name = "spirv_cube.branch_conditional",
    format,
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

#[op_interface_impl]
impl BranchOpFoldInterface for BranchConditionalOp {
    fn check_fold(&self, ctx: &Context, _operands: &[Option<AttrObj>]) -> Vec<Ptr<BasicBlock>> {
        self.get_operation().deref(ctx).successors().collect()
    }

    fn fold_in_place(
        &self,
        _ctx: &mut Context,
        _ops: &[Option<AttrObj>],
        _rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        IRStatus::Unchanged
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
        rewriter.append_op(ctx, &op);
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

        let branch_cond = BranchConditionalOp::new(
            ctx,
            self.condition(ctx),
            then_block,
            vec![],
            else_block,
            vec![],
        );
        rewriter.append_op(ctx, &branch_cond);

        rewriter.set_insertion_point_before_operation(then_term);
        if then_term.is_op::<YieldOp>(ctx) {
            let then_branch = BranchOp::new(ctx, merge, vec![]);
            rewriter.append_op(ctx, &then_branch);
            rewriter.erase_operation(ctx, then_term);
        } else if then_term.impls::<dyn IsExitTerminator>(ctx) {
            // Keep terminator, it's a valid terminator in SPIR-V
        } else {
            panic!("Unsupported terminator found in `IfOp`")
        }

        rewriter.inline_region(ctx, then_region, BlockInsertionPoint::AfterBlock(entry));

        if else_term.is_op::<YieldOp>(ctx) {
            rewriter.set_insertion_point_before_operation(else_term);
            let else_branch = BranchOp::new(ctx, merge, vec![]);
            rewriter.append_op(ctx, &else_branch);
            rewriter.erase_operation(ctx, else_term);
        } else if else_term.impls::<dyn IsExitTerminator>(ctx) {
            // Keep terminator, it's a valid terminator in SPIR-V
        } else {
            panic!("Unsupported terminator found in `IfOp`")
        }

        rewriter.inline_region(
            ctx,
            else_region,
            BlockInsertionPoint::AfterBlock(then_block),
        );

        rewriter.insert_block(ctx, BlockInsertionPoint::AtRegionEnd(select_region), merge);
        rewriter.set_insertion_point_to_block_end(merge);

        let merge_op = MergeOp::new(ctx, vec![]);
        rewriter.append_op(ctx, &merge_op);

        rewriter.set_insertion_point_before_operation(self.get_operation());
        rewriter.append_op(ctx, &selection);
        rewriter.replace_operation(ctx, self.get_operation(), selection.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvCFDialect for SwitchOp {
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

        let default_region = self.default_region(ctx);
        let default_block = self.default_block(ctx);
        let default_term = default_block.deref(ctx).get_terminator(ctx).unwrap();
        let cases = self.get_attr_branch_switch_cases(ctx).unwrap().clone();
        let case_dests = self.get_case_destinations(ctx);

        let merge = BasicBlock::new(ctx, None, vec![]);

        let switch = ops::SwitchOp::new(
            ctx,
            self.value(ctx),
            default_block,
            vec![],
            cases,
            case_dests.clone(),
            vec![Vec::new(); case_dests.len()],
        );
        rewriter.append_op(ctx, &switch);

        rewriter.set_insertion_point_before_operation(default_term);
        if default_term.is_op::<YieldOp>(ctx) {
            let default_branch = BranchOp::new(ctx, merge, vec![]);
            rewriter.append_op(ctx, &default_branch);
            rewriter.erase_operation(ctx, default_term);
        } else if default_term.impls::<dyn IsExitTerminator>(ctx) {
            // Keep terminator, it's a valid terminator in SPIR-V
        } else {
            panic!("Unsupported terminator found in `SwitchOp`")
        }

        rewriter.inline_region(ctx, default_region, BlockInsertionPoint::AfterBlock(entry));

        for case_dest in case_dests {
            let case_term = case_dest.deref(ctx).get_terminator(ctx).unwrap();
            if case_term.is_op::<YieldOp>(ctx) {
                rewriter.set_insertion_point_before_operation(case_term);
                let case_branch = BranchOp::new(ctx, merge, vec![]);
                rewriter.append_op(ctx, &case_branch);
                rewriter.erase_operation(ctx, case_term);
            } else if case_term.impls::<dyn IsExitTerminator>(ctx) {
                // Keep terminator, it's a valid terminator in SPIR-V
            } else {
                panic!("Unsupported terminator found in `SwitchOp`")
            }

            let region = case_dest.deref(ctx).get_parent_region().unwrap();
            rewriter.inline_region(ctx, region, BlockInsertionPoint::AtRegionEnd(select_region));
        }

        rewriter.insert_block(ctx, BlockInsertionPoint::AtRegionEnd(select_region), merge);
        rewriter.set_insertion_point_to_block_end(merge);

        let merge_op = MergeOp::new(ctx, vec![]);
        rewriter.append_op(ctx, &merge_op);

        rewriter.set_insertion_point_before_operation(self.get_operation());
        rewriter.append_op(ctx, &selection);
        rewriter.replace_operation(ctx, self.get_operation(), selection.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvCFDialect for RangeLoopOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        define_scalar!(I);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);

        let iter_var = self.iter_var(ctx);
        let start = self.start(ctx);
        let end = self.end(ctx);
        let step = self.step(ctx);

        scope.register_value_type::<I, ()>(start);

        let init = StoreOp::new(ctx, iter_var, start);
        rewriter.append_op(ctx, &init);

        let r#loop = LoopOp::new(ctx, vec![]);
        let loop_region = r#loop.region(ctx);
        let entry = r#loop.entry_block(ctx);
        rewriter.set_insertion_point_to_block_end(entry);

        let body_region = self.get_region(ctx);
        let body_block = self.loop_body(ctx);
        let body_term = body_block.deref(ctx).get_terminator(ctx).unwrap();

        let header = BasicBlock::new(ctx, None, vec![]);
        let header_branch = BranchOp::new(ctx, header, vec![]);
        rewriter.append_op(ctx, &header_branch);
        rewriter.insert_block(ctx, BlockInsertionPoint::AfterBlock(entry), header);
        rewriter.set_insertion_point_to_block_end(header);

        let merge = BasicBlock::new(ctx, None, vec![]);

        let iter_value = scope.register_with_result(&LoadOp::new(ctx, iter_var));
        let should_continue =
            I::__expand_native_lt(&scope, iter_value.into(), end.into()).read_value(&scope);

        let loop_branch =
            BranchConditionalOp::new(ctx, should_continue, body_block, vec![], merge, vec![]);
        rewriter.append_op(ctx, &loop_branch);

        rewriter.set_insertion_point_before_operation(body_term);
        let iter_value = scope.register_with_result(&LoadOp::new(ctx, iter_var));
        let next_value =
            I::__expand_native_add(&scope, iter_value.into(), step.into()).read_value(&scope);
        scope.register(&StoreOp::new(ctx, iter_var, next_value));

        if body_term.is_op::<YieldOp>(ctx) {
            let header_branch = BranchOp::new(ctx, header, vec![]);
            rewriter.append_op(ctx, &header_branch);
            rewriter.erase_operation(ctx, body_term);
        } else if body_term.impls::<dyn IsExitTerminator>(ctx) {
            // Keep terminator, it's a valid terminator in SPIR-V
        } else {
            panic!("Unsupported terminator found in `WhileOp`")
        }

        rewriter.inline_region(ctx, body_region, BlockInsertionPoint::AfterBlock(header));

        rewriter.insert_block(ctx, BlockInsertionPoint::AtRegionEnd(loop_region), merge);
        rewriter.set_insertion_point_to_block_end(merge);

        let merge_op = MergeOp::new(ctx, vec![]);
        rewriter.append_op(ctx, &merge_op);

        rewriter.set_insertion_point_before_operation(self.get_operation());
        rewriter.append_op(ctx, &r#loop);
        rewriter.replace_operation(ctx, self.get_operation(), r#loop.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvCFDialect for WhileOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let r#loop = LoopOp::new(ctx, vec![]);
        let loop_region = r#loop.region(ctx);
        let entry = r#loop.entry_block(ctx);
        rewriter.set_insertion_point_to_block_end(entry);

        let body_region = self.get_region(ctx);
        let body_block = self.loop_body(ctx);
        let body_term = body_block.deref(ctx).get_terminator(ctx).unwrap();

        let header = BasicBlock::new(ctx, None, vec![]);
        let header_branch = BranchOp::new(ctx, header, vec![]);
        rewriter.append_op(ctx, &header_branch);
        rewriter.insert_block(ctx, BlockInsertionPoint::AfterBlock(entry), header);
        rewriter.set_insertion_point_to_block_end(header);

        let merge = BasicBlock::new(ctx, None, vec![]);

        let load = LoadOp::new(ctx, self.cond_ptr(ctx));
        rewriter.append_op(ctx, &load);
        let loop_branch =
            BranchConditionalOp::new(ctx, load.get_result(ctx), body_block, vec![], merge, vec![]);
        rewriter.append_op(ctx, &loop_branch);

        if body_term.is_op::<YieldOp>(ctx) {
            rewriter.set_insertion_point_before_operation(body_term);
            let header_branch = BranchOp::new(ctx, header, vec![]);
            rewriter.append_op(ctx, &header_branch);
            rewriter.erase_operation(ctx, body_term);
        } else if body_term.impls::<dyn IsExitTerminator>(ctx) {
            // Keep terminator, it's a valid terminator in SPIR-V
        } else {
            panic!("Unsupported terminator found in `WhileOp`")
        }

        rewriter.inline_region(ctx, body_region, BlockInsertionPoint::AfterBlock(header));

        rewriter.insert_block(ctx, BlockInsertionPoint::AtRegionEnd(loop_region), merge);
        rewriter.set_insertion_point_to_block_end(merge);

        let merge_op = MergeOp::new(ctx, vec![]);
        rewriter.append_op(ctx, &merge_op);

        rewriter.set_insertion_point_before_operation(self.get_operation());
        rewriter.append_op(ctx, &r#loop);
        rewriter.replace_operation(ctx, self.get_operation(), r#loop.get_operation());

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
        rewriter.append_op(ctx, &return_);
        rewriter.replace_operation(ctx, op, return_.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvCFDialect for branch::UnreachableOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let unreachable = ops::UnreachableOp::new(ctx);
        rewriter.append_op(ctx, &unreachable);
        rewriter.replace_operation(ctx, op, unreachable.get_operation());

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
