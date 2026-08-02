//! Lower level branch dialect for targets that support block args/phi. Separate from the base
//! dialect because lifting back out of this is too hard for C++/WGSL. This can be used for
//! optimizations that require structured control flow but can also support transformations like
//! `mem2reg` or PRE that require threading SSA values in and out of the control flow ops.
//!
//! Most terminators are reused from `branch`, only `LoopYieldOp` is new.

use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::attributes::{IntegerAttr, VecAttr},
    irbuild::inserter::OpInsertionPoint,
    linked_list::ContainsLinkedList,
    opts::{constants::ConstFoldInterface, dce::SideEffects},
    region::Region,
    verify_err,
};

use crate::{
    CanMaterialize, NoMemoryEffect,
    attributes::BoolAttr,
    dialect::branch::{self, DeadRegionOp, YieldOp, YieldOpVerifyErr, block_side_effects},
    prelude::*,
    types::{PointerType, scalar::BoolType},
};

#[pliron_op(name = "scf.loop_yield", format)]
#[op_interfaces(IsTerminatorInterface, OperandSegmentInterface)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct LoopYieldOp;

impl LoopYieldOp {
    pub fn new(ctx: &mut Context, continue_args: Vec<Value>, exit_args: Vec<Value>) -> Self {
        let (operands, segment_sizes) = Self::compute_segment_sizes(vec![continue_args, exit_args]);

        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![],
                0,
            ),
        };
        op.set_operand_segment_sizes(ctx, segment_sizes);
        op
    }

    pub fn continue_args(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 0)
    }

    pub fn exit_args(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }
}

impl Verify for LoopYieldOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(parent_op) = self.get_operation().deref(ctx).get_parent_op(ctx) else {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::MissingParentOp);
        };

        let exit_expected_types: Vec<_> = parent_op
            .deref(ctx)
            .results()
            .map(|r| r.get_type(ctx))
            .collect();
        let exit_actual_types: Vec<_> = self
            .get_operation()
            .deref(ctx)
            .operands()
            .map(|o| o.get_type(ctx))
            .collect();

        if exit_expected_types != exit_actual_types {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::OperandTypeMismatch);
        }

        Ok(())
    }
}

#[pliron_op(
    name = "scf.if",
    format = "$0 ` then ` region($0) ` else ` region($1)",
    verifier = "succ"
)]
#[op_interfaces(NOpdsInterface<1>, NRegionsInterface<2>, SingleBlockRegionInterface, OperandNOfType<0, BoolType>)]
pub struct IfOp;

impl IfOp {
    pub fn new(ctx: &mut Context, results: Vec<TypeHandle>, cond: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            vec![cond],
            vec![],
            2,
        );

        let then_region = op.deref_mut(ctx).get_region(0);
        let then_body = BasicBlock::new(ctx, Some("then".try_into().unwrap()), vec![]);
        then_body.insert_at_front(then_region, ctx);

        let else_region = op.deref_mut(ctx).get_region(1);
        let else_body = BasicBlock::new(ctx, Some("else".try_into().unwrap()), vec![]);
        else_body.insert_at_front(else_region, ctx);

        Self { op }
    }

    pub fn condition(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn then_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn then_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }

    pub fn else_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }

    pub fn else_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 1)
    }

    pub fn results(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().results(ctx)
    }

    pub fn result_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        self.get_operation().result_types(ctx)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for IfOp {
    fn check_fold(
        &self,
        _ctx: &Context,
        operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        operand_attrs.to_vec()
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        let op = self.get_operation();
        let Some(attr) = operand_attrs[0].as_ref() else {
            return IRStatus::Unchanged;
        };
        let Some(attr) = attr.downcast_ref::<BoolAttr>() else {
            return IRStatus::Unchanged;
        };
        let (taken, not_taken) = match attr.0 {
            true => (self.then_block(ctx), self.else_block(ctx)),
            false => (self.else_block(ctx), self.then_block(ctx)),
        };

        let not_taken_op = DeadRegionOp::new(ctx);
        let dead_block = not_taken_op.get_body(ctx, 0);
        rewriter.append_op(ctx, &not_taken_op);

        let term = taken.deref(ctx).get_terminator(ctx);

        if let Some(term) = term
            && term.is_op::<YieldOp>(ctx)
        {
            let results = self.results(ctx);
            let yielded = term.operands(ctx);
            assert_eq!(results.len(), yielded.len(), "Yield doesn't match results");
            for (res, yielded) in results.into_iter().zip(yielded) {
                rewriter.replace_value_uses_with(ctx, res, yielded);
            }
        }

        inline_block(ctx, rewriter, taken, OpInsertionPoint::BeforeOperation(op));
        inline_block(
            ctx,
            rewriter,
            not_taken,
            OpInsertionPoint::AtBlockStart(dead_block),
        );

        IRStatus::Changed
    }
}

fn inline_block(
    ctx: &Context,
    rewriter: &mut dyn Rewriter,
    block: Ptr<BasicBlock>,
    insertion_point: OpInsertionPoint,
) {
    let ops = block.deref(ctx).iter(ctx).collect::<Vec<_>>();
    let mut insertion_pt = insertion_point;
    for op in ops {
        if !op.is_terminator(ctx) {
            rewriter.move_operation(ctx, op, insertion_pt);
            insertion_pt = OpInsertionPoint::AfterOperation(op);
        }
    }
}

#[op_interface_impl]
impl SideEffects for IfOp {
    fn has_side_effects(&self, ctx: &Context) -> bool {
        block_side_effects(ctx, self.then_block(ctx))
            || block_side_effects(ctx, self.else_block(ctx))
    }
}

#[op_interface_impl]
impl BranchToSCFOp for branch::IfOp {
    fn to_scf(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let opds = self.get_operation().operands(ctx);
        let op = Operation::new(ctx, IfOp::get_concrete_op_info(), vec![], opds, vec![], 0);

        let regions = self.get_operation().regions(ctx);
        for region in regions {
            Region::move_to_op(region, op, ctx);
        }

        rewriter.append_operation(ctx, op);
        rewriter.replace_operation(ctx, self.get_operation(), op);
        Ok(())
    }
}

#[pliron_op(
    name = "scf.switch",
    format,
    attributes = (scf_switch_cases: VecAttr),
    verifier = "succ"
)]
#[op_interfaces(NOpdsInterface<1>, SingleBlockRegionInterface)]
pub struct SwitchOp;

impl SwitchOp {
    pub fn new(ctx: &mut Context, results: Vec<TypeHandle>, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            vec![value],
            vec![],
            1,
        );

        let default_region = op.deref_mut(ctx).get_region(0);
        let default_body = BasicBlock::new(ctx, Some("default".try_into().unwrap()), vec![]);
        default_body.insert_at_front(default_region, ctx);

        Self { op }
    }

    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn default_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn default_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }

    pub fn append_case_block(&self, ctx: &mut Context) -> Ptr<BasicBlock> {
        let region = Operation::add_region(self.get_operation(), ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_front(region, ctx);
        region.deref(ctx).get_head().unwrap()
    }

    pub fn cases(&self, ctx: &Context) -> Vec<(IntegerAttr, Ptr<BasicBlock>)> {
        let cases = self.get_attr_scf_switch_cases(ctx).unwrap().clone().0;
        let out = (0..cases.len()).map(|i| {
            let value = cases[i].downcast_ref::<IntegerAttr>().unwrap().clone();
            let block = self.get_body(ctx, i + 1);
            (value, block)
        });
        out.collect()
    }

    pub fn get_case_destinations(&self, ctx: &Context) -> Vec<Ptr<BasicBlock>> {
        let op = self.get_operation().deref(ctx);
        (1..op.regions().count())
            .map(|i| self.get_body(ctx, i))
            .collect()
    }

    pub fn set_attr_cases(&self, ctx: &Context, cases: impl IntoIterator<Item = AttrObj>) {
        self.set_attr_scf_switch_cases(ctx, VecAttr(cases.into_iter().collect()));
    }

    pub fn results(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().results(ctx)
    }

    pub fn result_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        self.get_operation().result_types(ctx)
    }
}

#[op_interface_impl]
impl BranchToSCFOp for branch::SwitchOp {
    fn to_scf(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let opds = self.get_operation().operands(ctx);
        let info = SwitchOp::get_concrete_op_info();
        let op = Operation::new(ctx, info, vec![], opds, vec![], 0);
        let cases = self.get_attr_branch_switch_cases(ctx).unwrap().clone();
        SwitchOp { op }.set_attr_scf_switch_cases(ctx, cases);

        let regions = self.get_operation().regions(ctx);
        for region in regions {
            Region::move_to_op(region, op, ctx);
        }

        rewriter.append_operation(ctx, op);
        rewriter.replace_operation(ctx, self.get_operation(), op);
        Ok(())
    }
}

#[pliron_op(
    name = "scf.range_loop",
    format = "`for ` $0 ` = ` $1 ` to ` $2 ` step ` $3 ` do ` region($0)",
    verifier = "succ"
)]
#[op_interfaces(OneRegionInterface, SingleBlockRegionInterface)]
pub struct RangeLoopOp;

impl RangeLoopOp {
    pub fn new(
        ctx: &mut Context,
        results: Vec<TypeHandle>,
        iter_var: Value,
        start: Value,
        end: Value,
        step: Value,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            vec![iter_var, start, end, step],
            vec![],
            1,
        );

        let body_region = op.deref_mut(ctx).get_region(0);
        let body = BasicBlock::new(ctx, Some("body".try_into().unwrap()), vec![]);
        body.insert_at_front(body_region, ctx);

        Self { op }
    }

    pub fn iter_var(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn start(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn end(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    pub fn step(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(3)
    }

    pub fn loop_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn loop_body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }

    pub fn results(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().results(ctx)
    }

    pub fn result_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        self.get_operation().result_types(ctx)
    }
}

#[pliron_op(
    name = "scf.while",
    format = "`*`$0 ` do ` region($0)",
    verifier = "succ"
)]
#[op_interfaces(
    OperandNOfType<0, PointerType>,
    OneRegionInterface,
    SingleBlockRegionInterface
)]
pub struct WhileOp;

impl WhileOp {
    pub fn new(ctx: &mut Context, results: Vec<TypeHandle>, cond_ptr: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            vec![cond_ptr],
            vec![],
            1,
        );

        let body_region = op.deref_mut(ctx).get_region(0);
        let body = BasicBlock::new(ctx, Some("body".try_into().unwrap()), vec![]);
        body.insert_at_front(body_region, ctx);

        Self { op }
    }

    pub fn cond_ptr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn loop_body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }

    pub fn results(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().results(ctx)
    }

    pub fn result_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        self.get_operation().result_types(ctx)
    }
}

#[op_interface]
trait BranchToSCFOp {
    verify_op_succ!();
    fn to_scf(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}

pub type BranchToSCFPass = DialectConversionPass<BranchToSCFConversion>;

#[derive(Default)]
pub struct BranchToSCFConversion;

impl DialectConversion for BranchToSCFConversion {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn BranchToSCFOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let to_scf = op_cast::<dyn BranchToSCFOp>(&*dyn_op).unwrap();
        to_scf.to_scf(ctx, rewriter, operands_info)
    }
}

macro_rules! loop_to_scf {
    ($branch_ty: ty, $scf: ty) => {
        #[op_interface_impl]
        impl BranchToSCFOp for $branch_ty {
            fn to_scf(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _: &OperandsInfo,
            ) -> Result<()> {
                let opds = self.get_operation().operands(ctx);
                let op =
                    Operation::new(ctx, <$scf>::get_concrete_op_info(), vec![], opds, vec![], 0);

                let body = self.loop_body(ctx);
                let Some(term) = body.deref(ctx).get_terminator(ctx) else {
                    return verify_err!(self.loc(ctx), "Should have terminator in loop body");
                };

                let regions = self.get_operation().regions(ctx);
                for region in regions {
                    Region::move_to_op(region, op, ctx);
                }

                rewriter.append_operation(ctx, op);
                rewriter.replace_operation(ctx, self.get_operation(), op);

                rewriter.set_insertion_point_before_operation(term);

                if term.is_op::<YieldOp>(ctx) {
                    let new_yield = LoopYieldOp::new(ctx, vec![], vec![]);
                    rewriter.append_op(ctx, &new_yield);
                    rewriter.replace_operation(ctx, term, new_yield.get_operation());
                }

                Ok(())
            }
        }
    };
}

loop_to_scf!(branch::RangeLoopOp, RangeLoopOp);
loop_to_scf!(branch::WhileOp, WhileOp);
