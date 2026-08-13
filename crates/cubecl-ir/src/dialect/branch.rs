use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::attributes::{IntegerAttr, VecAttr},
    irbuild::inserter::OpInsertionPoint,
    linked_list::ContainsLinkedList,
    opts::{constants::ConstFoldInterface, dce::SideEffects},
    region::Region,
    utils::const_bound_n::I,
    verify_err,
};
use thiserror::Error;

use crate::{
    CanMaterialize, NoMemoryEffect, Pure,
    attributes::{BoolAttr, ZeroAttr},
    prelude::*,
    types::scalar::BoolType,
};

/// Marker for terminators that do not return, i.e. `UnreachableOp`. In Rust terms, they return `!`.
#[op_interface]
pub trait IsExitTerminator {
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum YieldOpVerifyErr {
    #[error("YieldOp operand types do not match parent operation result types")]
    OperandTypeMismatch,
    #[error("YieldOp must have a parent operation to verify against")]
    MissingParentOp,
}

#[pliron_op(name = "branch.yield", format = "`(` operands(CharSpace(`,`)) `)`")]
#[op_interfaces(IsTerminatorInterface)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct YieldOp;

impl YieldOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
    }

    pub fn yield_values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().operands(ctx)
    }
}

impl Verify for YieldOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(parent_op) = self.get_operation().deref(ctx).get_parent_op(ctx) else {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::MissingParentOp);
        };

        let expected_types: Vec<_> = parent_op
            .deref(ctx)
            .results()
            .map(|r| r.get_type(ctx))
            .collect();
        let actual_types: Vec<_> = self
            .get_operation()
            .deref(ctx)
            .operands()
            .map(|o| o.get_type(ctx))
            .collect();

        if expected_types != actual_types {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::OperandTypeMismatch);
        }

        Ok(())
    }
}

#[pliron_op(name = "branch.condition", format = "`(` operands(CharSpace(`,`)) `)`")]
#[op_interfaces(IsTerminatorInterface, OperandNOfType<0, BoolType>)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ConditionOp;

impl ConditionOp {
    pub fn new(ctx: &mut Context, cond: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![cond],
            vec![],
            0,
        );
        Self { op }
    }

    pub fn condition(&self, ctx: &Context) -> Value {
        self.get_operation().operand(ctx, 0)
    }

    pub fn forward_values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}

impl Verify for ConditionOp {
    fn verify(&self, ctx: &Context) -> pliron::result::Result<()> {
        let Some(parent_op) = self.get_operation().deref(ctx).get_parent_op(ctx) else {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::MissingParentOp);
        };

        let expected_types: Vec<_> = parent_op
            .deref(ctx)
            .results()
            .map(|r| r.get_type(ctx))
            .collect();
        let actual_types: Vec<_> = self
            .forward_values(ctx)
            .into_iter()
            .map(|o| o.get_type(ctx))
            .collect();

        if expected_types != actual_types {
            return verify_err!(self.loc(ctx), YieldOpVerifyErr::OperandTypeMismatch);
        }

        Ok(())
    }
}

#[pliron_op(
    name = "branch.return",
    format = "operands(CharSpace(`,`))",
    verifier = "succ"
)]
#[op_interfaces(IsTerminatorInterface, IsExitTerminator)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ReturnOp;

impl ReturnOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
    }

    pub fn new_with_value(ctx: &mut Context, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        Self { op }
    }

    pub fn value(&self, ctx: &Context) -> Option<Value> {
        self.get_operation().deref(ctx).results().next()
    }
}

#[pliron_op(name = "branch.unreachable", format = "", verifier = "succ")]
#[op_interfaces(IsTerminatorInterface, IsExitTerminator)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct UnreachableOp;

impl UnreachableOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
    }
}

/// Dead region for constant folding, returns a dummy result so it gets eliminated from dead code
/// elimination. We can't erase the block straight away because it might contain SCCP candidates
/// that are already tracked and will cause a dangling ptr deref.
#[pliron_op(name = "branch.dead_region", format = "region($0)", verifier = "succ")]
#[op_interfaces(NOpdsInterface<0>, OneResultInterface, OneRegionInterface, SingleBlockRegionInterface)]
#[op_traits(Pure)]
pub struct DeadRegionOp;

impl DeadRegionOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);

        let region = op.deref_mut(ctx).get_region(0);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_front(region, ctx);

        Self { op }
    }

    pub fn region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }
}

pub(super) fn block_side_effects(ctx: &Context, block: Ptr<BasicBlock>) -> bool {
    block.deref(ctx).iter(ctx).any(|op| {
        // Yield should not count as an effect in a region, but also can't implement
        // `SideEffects = true` because then it would immediately get eliminated
        if op.is_op::<YieldOp>(ctx) {
            return false;
        }
        match op_cast::<dyn SideEffects>(&*op.dyn_op(ctx)) {
            Some(side_effects) => side_effects.has_side_effects(ctx),
            None => true,
        }
    })
}

#[pliron_op(
    name = "branch.if",
    format = "$0 ` then ` region($0) ` else ` region($1)",
    verifier = "succ"
)]
#[op_interfaces(NOpdsInterface<1>, NResultsInterface<0>, NRegionsInterface<2>, SingleBlockRegionInterface, OperandNOfType<0, BoolType>)]
pub struct IfOp;

impl IfOp {
    pub fn new(ctx: &mut Context, cond: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
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
        let zero = attr.downcast_ref::<ZeroAttr>().map(|_| false);
        let bool = attr.downcast_ref::<BoolAttr>().map(|it| it.0);
        let Some(const_cond) = zero.or(bool) else {
            return IRStatus::Unchanged;
        };
        let (taken, not_taken) = match const_cond {
            true => (self.then_block(ctx), self.else_block(ctx)),
            false => (self.else_block(ctx), self.then_block(ctx)),
        };

        let not_taken_op = DeadRegionOp::new(ctx);
        let dead_block = not_taken_op.get_body(ctx, 0);
        rewriter.append_op(ctx, &not_taken_op);

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

#[pliron_op(
    name = "branch.switch",
    format,
    attributes = (branch_switch_cases: VecAttr),
    verifier = "succ"
)]
#[op_interfaces(NOpdsInterface<1>, NResultsInterface<0>, SingleBlockRegionInterface)]
pub struct SwitchOp;

impl SwitchOp {
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
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
        let cases = self.get_attr_branch_switch_cases(ctx).unwrap().clone().0;
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
        self.set_attr_branch_switch_cases(ctx, VecAttr(cases.into_iter().collect()));
    }
}

#[pliron_op(name = "branch.range_loop", format, verifier = "succ")]
#[op_interfaces(NResultsInterface<0>, OneRegionInterface, SingleBlockRegionInterface, SameOperandsType)]
pub struct RangeLoopOp;

impl RangeLoopOp {
    pub fn new(ctx: &mut Context, start: Value, end: Value, step: Value) -> Self {
        let iter_ty = start.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![start, end, step],
            vec![],
            1,
        );

        let body_region = op.deref_mut(ctx).get_region(0);
        let body = BasicBlock::new(ctx, Some("body".try_into().unwrap()), vec![iter_ty]);
        body.insert_at_front(body_region, ctx);

        Self { op }
    }

    pub fn iter_var(&self, ctx: &Context) -> Value {
        self.loop_body(ctx).deref(ctx).get_argument(0)
    }

    pub fn start(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn end(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    pub fn step(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    pub fn loop_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn loop_body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }
}

#[pliron_op(
    name = "branch.while",
    format = "`while ` region($0) ` do ` region($1)",
    verifier = "succ"
)]
#[op_interfaces(
    NResultsInterface<0>,
    NRegionsInterface<2>,
    SingleBlockRegionInterface
)]
pub struct WhileOp;

impl WhileOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 2);

        let before_region = op.deref_mut(ctx).get_region(0);
        let before = BasicBlock::new(ctx, Some("before".try_into().unwrap()), vec![]);
        before.insert_at_front(before_region, ctx);

        let after_region = op.deref_mut(ctx).get_region(1);
        let after = BasicBlock::new(ctx, Some("after".try_into().unwrap()), vec![]);
        after.insert_at_front(after_region, ctx);

        Self { op }
    }

    pub fn before_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_region_i(ctx, I::<0>.into())
    }

    pub fn before_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }

    pub fn after_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_region_i(ctx, I::<1>.into())
    }

    pub fn after_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 1)
    }
}
