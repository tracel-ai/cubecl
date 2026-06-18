use pliron::{
    basic_block::BasicBlock, builtin::attributes::VecAttr, linked_list::ContainsLinkedList,
    region::Region, verify_err,
};
use thiserror::Error;

use crate::{
    attributes::IntAttr,
    prelude::*,
    types::{
        PointerType,
        scalar::{BoolType, IndexType},
    },
};

#[derive(Error, Debug)]
pub enum YieldOpVerifyErr {
    #[error("YieldOp operand types do not match parent operation result types")]
    OperandTypeMismatch,
    #[error("YieldOp must have a parent operation to verify against")]
    MissingParentOp,
}

#[pliron_op(name = "branch.yield", format)]
#[op_interfaces(IsTerminatorInterface)]
pub struct YieldOp;

impl YieldOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
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

#[pliron_op(name = "branch.return", format, verifier = "succ")]
#[op_interfaces(IsTerminatorInterface)]
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

#[pliron_op(name = "branch.break", format = "$0", verifier = "succ")]
#[op_interfaces(IsTerminatorInterface)]
pub struct BreakOp;

impl BreakOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
    }
}

#[pliron_op(name = "branch.unreachable", format = "$0", verifier = "succ")]
#[op_interfaces(IsTerminatorInterface)]
pub struct UnreachableOp;

impl UnreachableOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Self { op }
    }
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

#[pliron_op(
    name = "branch.switch",
    format,
    attributes = (cases: VecAttr),
    verifier = "succ"
)]
#[op_interfaces(NOpdsInterface<1>, NResultsInterface<0>, NRegionsInterface<2>, SingleBlockRegionInterface, OperandNOfType<0, IndexType>)]
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

    pub fn cases(&self, ctx: &Context) -> Vec<(IntAttr, Ptr<BasicBlock>)> {
        let cases = self.get_attr_cases(ctx).unwrap().clone().0;
        let out = (0..cases.len()).map(|i| {
            let value = *cases[i].downcast_ref::<IntAttr>().unwrap();
            let block = self.get_body(ctx, i);
            (value, block)
        });
        out.collect()
    }
}

#[pliron_op(
    name = "branch.range_loop",
    format = "`for *`$0 ` = ` $1 ` to ` $2 ` step ` $3 ` do ` region($0)",
    verifier = "succ"
)]
#[op_interfaces(NResultsInterface<0>, NRegionsInterface<1>, SingleBlockRegionInterface)]
pub struct RangeLoopOp;

impl RangeLoopOp {
    pub fn new(ctx: &mut Context, iter_var: Value, start: Value, end: Value, step: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
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
}

#[pliron_op(
    name = "branch.while",
    format = "`*`$0 ` do ` region($0)",
    verifier = "succ"
)]
#[op_interfaces(
    OperandNOfType<0, PointerType>,
    NResultsInterface<0>,
    NRegionsInterface<1>,
    SingleBlockRegionInterface
)]
pub struct WhileOp;

impl WhileOp {
    pub fn new(ctx: &mut Context, cond_ptr: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
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
}

#[pliron_op(name = "branch.loop", format = "`loop ` region($0)", verifier = "succ")]
#[op_interfaces(
    NOpdsInterface<1>,
    NResultsInterface<0>,
    NRegionsInterface<1>,
    SingleBlockRegionInterface,
)]
pub struct LoopOp;

impl LoopOp {
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);

        let body_region = op.deref_mut(ctx).get_region(0);
        let body = BasicBlock::new(ctx, Some("body".try_into().unwrap()), vec![]);
        body.insert_at_front(body_region, ctx);

        Self { op }
    }

    pub fn loop_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn loop_body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }
}
