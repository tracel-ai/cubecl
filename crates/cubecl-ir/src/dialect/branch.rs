use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::attributes::IntegerAttr,
    irbuild::inserter::OpInsertionPoint,
    linked_list::ContainsLinkedList,
    opts::dce::SideEffects,
    region::Region,
    utils::{
        const_bound_n::I,
        table::{HMap, SmallMap},
    },
    verify_err,
};
use thiserror::Error;

use crate::{
    CanMaterialize, NoMemoryEffect, ReturnLike,
    attributes::{BoolAttr, IntegerVecAttr, ZeroAttr},
    dialect::scf::block_mem_val,
    interfaces::{
        CanonicalizeInterface,
        control_flow::{
            InvocationBounds, RegionBranchOpInterface, RegionBranchTerminatorOpInterface,
            RegionPredecessor, RegionSuccessor,
        },
        memory_slot::{
            MemoryRegionPredecessor, MemorySSAContext, MemorySSARegionOpInterface, MemoryValue,
            RegionMemoryPhiInputs, RegionMemoryValue,
        },
        uniformity::{UniformRegionTerminatorOpInterface, Uniformity},
    },
    prelude::*,
    small_map,
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
#[op_interfaces(IsTerminatorInterface, NResultsInterface<0>)]
#[op_traits(NoMemoryEffect, ReturnLike, CanMaterialize)]
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
#[op_interfaces(IsTerminatorInterface, NResultsInterface<0>, OperandNOfType<0, BoolType>)]
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

#[op_interface_impl]
impl RegionBranchTerminatorOpInterface for ConditionOp {
    fn successor_operands(&self, ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        self.forward_values(ctx)
    }

    fn successor_regions(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let while_op = self.get_operation().deref(ctx).get_parent_op(ctx).unwrap();
        let after_region = while_op.deref(ctx).get_region(1).into();
        let Some(attr) = operands[0].as_ref() else {
            return vec![after_region, RegionSuccessor::AfterOp];
        };
        let zero = attr.downcast_ref::<ZeroAttr>().map(|_| false);
        let bool = attr.downcast_ref::<BoolAttr>().map(|it| it.0);
        let Some(const_cond) = zero.or(bool) else {
            return vec![after_region, RegionSuccessor::AfterOp];
        };
        match const_cond {
            true => vec![after_region],
            false => vec![RegionSuccessor::AfterOp],
        }
    }
}

#[op_interface_impl]
impl UniformRegionTerminatorOpInterface for ConditionOp {
    fn successor_region_uniformity(
        &self,
        ctx: &Context,
        operands: &[Uniformity],
    ) -> Vec<Uniformity> {
        self.all_successor_regions(ctx)
            .iter()
            .map(|_| operands[0])
            .collect()
    }
}

#[pliron_op(
    name = "branch.return",
    format = "operands(CharSpace(`,`))",
    verifier = "succ"
)]
#[op_interfaces(IsTerminatorInterface, NResultsInterface<0>, IsExitTerminator)]
#[op_traits(ReturnLike, CanMaterialize, NoMemoryEffect)]
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
impl MemorySSARegionOpInterface for IfOp {
    fn setup_memory_ssa(
        &self,
        ctx: &Context,
        _state: &mut MemorySSAContext,
        reaching_def: MemoryValue,
        _has_memory_defs: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, MemoryValue, 2>,
    ) {
        regions_to_process.insert(self.then_region(ctx), reaching_def);
        regions_to_process.insert(self.else_region(ctx), reaching_def);
    }

    fn finalize_memory_ssa(
        &self,
        ctx: &Context,
        _state: &mut MemorySSAContext,
        entry_reaching_def: MemoryValue,
        has_memory_defs: bool,
        _reaching_at_region_entry: &HMap<Ptr<Region>, MemoryValue>,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, MemoryValue>,
        _region_phis: &mut SmallMap<Ptr<Region>, RegionMemoryPhiInputs, 2>,
    ) -> RegionMemoryValue {
        if !has_memory_defs {
            return RegionMemoryValue::Forward(entry_reaching_def);
        }

        let (then_pred, reaching_then) = block_mem_val(
            self.then_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );
        let (else_pred, reaching_else) = block_mem_val(
            self.else_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );

        RegionMemoryValue::RegionPhi(small_map! {
            then_pred => reaching_then,
            else_pred => reaching_else
        })
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for IfOp {
    fn entry_successor_regions(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let Some(attr) = operands[0].as_ref() else {
            return self.successor_regions(ctx, RegionPredecessor::Parent);
        };
        let zero = attr.downcast_ref::<ZeroAttr>().map(|_| false);
        let bool = attr.downcast_ref::<BoolAttr>().map(|it| it.0);
        let Some(const_cond) = zero.or(bool) else {
            return self.successor_regions(ctx, RegionPredecessor::Parent);
        };
        match const_cond {
            true => vec![self.then_region(ctx).into()],
            false => vec![self.else_region(ctx).into()],
        }
    }

    fn successor_regions(&self, ctx: &Context, pred: RegionPredecessor) -> Vec<RegionSuccessor> {
        match pred {
            RegionPredecessor::Parent => {
                vec![self.then_region(ctx).into(), self.else_region(ctx).into()]
            }
            RegionPredecessor::Terminator(_) => {
                vec![RegionSuccessor::AfterOp]
            }
        }
    }

    fn successor_inputs(&self, _ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        vec![]
    }

    fn region_invocation_bounds(
        &self,
        _ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<InvocationBounds> {
        if let Some(cond) = operands[0]
            .as_ref()
            .and_then(|it| it.downcast_ref::<BoolAttr>())
        {
            match cond.0 {
                true => vec![InvocationBounds::once(), InvocationBounds::never()],
                false => vec![InvocationBounds::never(), InvocationBounds::once()],
            }
        } else {
            vec![InvocationBounds::zero_or_one(); 2]
        }
    }
}

impl IfOp {
    fn fold(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let op = self.get_operation();
        let operands = const_operands(ctx, op);
        let valid_branches = self.entry_successor_regions(ctx, &operands);
        let &[RegionSuccessor::Region(taken)] = valid_branches.as_slice() else {
            return Ok(());
        };
        let taken = taken.deref(ctx).get_entry_block().unwrap();

        inline_block(ctx, rewriter, taken, OpInsertionPoint::BeforeOperation(op));
        rewriter.erase_operation(ctx, op);

        Ok(())
    }
}

#[op_interface_impl]
impl CanonicalizeInterface for IfOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        self.fold(ctx, rewriter)?;
        Ok(())
    }
}

#[pliron_op(
    name = "branch.switch",
    format,
    attributes = (branch_switch_cases: IntegerVecAttr),
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
            let value = cases[i].clone();
            let block = self.get_body(ctx, i + 1);
            (value, block)
        });
        out.collect()
    }

    pub fn cases_regions(&self, ctx: &Context) -> Vec<(IntegerAttr, Ptr<Region>)> {
        let cases = self.get_attr_branch_switch_cases(ctx).unwrap().clone().0;
        let out = (0..cases.len()).map(|i| {
            let value = cases[i].clone();
            let block = self.get_operation().deref(ctx).get_region(i + 1);
            (value, block)
        });
        out.collect()
    }

    pub fn cases_values(&self, ctx: &Context) -> Vec<IntegerAttr> {
        self.get_attr_branch_switch_cases(ctx).unwrap().0.clone()
    }

    pub fn get_case_destinations(&self, ctx: &Context) -> Vec<Ptr<BasicBlock>> {
        let op = self.get_operation().deref(ctx);
        (1..op.regions().count())
            .map(|i| self.get_body(ctx, i))
            .collect()
    }

    pub fn set_attr_cases(&self, ctx: &Context, cases: impl IntoIterator<Item = IntegerAttr>) {
        self.set_attr_branch_switch_cases(ctx, IntegerVecAttr(cases.into_iter().collect()));
    }
}

#[op_interface_impl]
impl MemorySSARegionOpInterface for SwitchOp {
    fn setup_memory_ssa(
        &self,
        ctx: &Context,
        _state: &mut MemorySSAContext,
        reaching_def: MemoryValue,
        _has_memory_defs: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, MemoryValue, 2>,
    ) {
        regions_to_process.insert(self.default_region(ctx), reaching_def);
        for (_, case_region) in self.cases_regions(ctx) {
            regions_to_process.insert(case_region, reaching_def);
        }
    }

    fn finalize_memory_ssa(
        &self,
        ctx: &Context,
        _state: &mut MemorySSAContext,
        entry_reaching_def: MemoryValue,
        has_memory_defs: bool,
        _reaching_at_region_entry: &HMap<Ptr<Region>, MemoryValue>,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, MemoryValue>,
        _region_phis: &mut SmallMap<Ptr<Region>, RegionMemoryPhiInputs, 2>,
    ) -> RegionMemoryValue {
        if !has_memory_defs {
            return RegionMemoryValue::Forward(entry_reaching_def);
        }

        let mut phi_inputs = SmallMap::new();

        let (default_pred, reaching_default) = block_mem_val(
            self.default_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );
        phi_inputs.insert(default_pred, reaching_default);

        for (_, case_block) in self.cases(ctx) {
            let (case_pred, reaching_case) =
                block_mem_val(case_block, entry_reaching_def, reaching_at_block_end);
            phi_inputs.insert(case_pred, reaching_case);
        }

        RegionMemoryValue::RegionPhi(phi_inputs)
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for SwitchOp {
    fn entry_successor_regions(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<RegionSuccessor> {
        let Some(attr) = &operands[0] else {
            return self.successor_regions(ctx, RegionPredecessor::Parent);
        };
        let Some(attr) = attr.downcast_ref::<IntegerAttr>() else {
            return self.successor_regions(ctx, RegionPredecessor::Parent);
        };
        if let Some(&(_, case)) = self.cases_regions(ctx).iter().find(|(val, _)| val == attr) {
            vec![case.into()]
        } else {
            vec![self.default_region(ctx).into()]
        }
    }

    fn successor_regions(&self, ctx: &Context, pred: RegionPredecessor) -> Vec<RegionSuccessor> {
        match pred {
            RegionPredecessor::Parent => {
                let op = self.get_operation().deref(ctx);
                op.regions().map(Into::into).collect()
            }
            RegionPredecessor::Terminator(_) => {
                vec![RegionSuccessor::AfterOp]
            }
        }
    }

    fn successor_inputs(&self, _ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        vec![]
    }

    fn region_invocation_bounds(
        &self,
        ctx: &Context,
        operands: &[Option<AttrObj>],
    ) -> Vec<InvocationBounds> {
        let num_regions = self.get_operation().deref(ctx).num_regions();
        let Some(attr) = operands[0].as_ref() else {
            return vec![InvocationBounds::zero_or_one(); num_regions];
        };
        let Some(attr) = attr.downcast_ref::<IntegerAttr>() else {
            return vec![InvocationBounds::zero_or_one(); num_regions];
        };

        let case_idx = self.cases_values(ctx).iter().position(|it| it == attr);
        let executed_idx = case_idx.map(|i| i + 1).unwrap_or(0);
        let mut bounds = vec![InvocationBounds::never(); num_regions];
        bounds[executed_idx] = InvocationBounds::once();
        bounds
    }
}

impl SwitchOp {
    fn fold(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let op = self.get_operation();
        let operands = const_operands(ctx, op);
        let valid_branches = self.entry_successor_regions(ctx, &operands);
        let &[RegionSuccessor::Region(taken)] = valid_branches.as_slice() else {
            return Ok(());
        };
        let taken = taken.deref(ctx).get_entry_block().unwrap();

        inline_block(ctx, rewriter, taken, OpInsertionPoint::BeforeOperation(op));
        rewriter.erase_operation(ctx, op);

        Ok(())
    }
}

#[op_interface_impl]
impl CanonicalizeInterface for SwitchOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        self.fold(ctx, rewriter)?;
        Ok(())
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

#[op_interface_impl]
impl MemorySSARegionOpInterface for RangeLoopOp {
    fn setup_memory_ssa(
        &self,
        ctx: &Context,
        state: &mut MemorySSAContext,
        reaching_def: MemoryValue,
        has_memory_defs: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, MemoryValue, 2>,
    ) {
        let body_region = self.loop_region(ctx);
        if !has_memory_defs {
            regions_to_process.insert(body_region, reaching_def);
            return;
        }

        let new_arg = state.new_value_in_block(self.loop_body(ctx));
        regions_to_process.insert(body_region, new_arg);
    }

    fn finalize_memory_ssa(
        &self,
        ctx: &Context,
        _state: &mut MemorySSAContext,
        entry_reaching_def: MemoryValue,
        has_memory_defs: bool,
        _reaching_at_region_entry: &HMap<Ptr<Region>, MemoryValue>,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, MemoryValue>,
        region_phis: &mut SmallMap<Ptr<Region>, RegionMemoryPhiInputs, 2>,
    ) -> RegionMemoryValue {
        let body_region = self.loop_region(ctx);
        if !has_memory_defs {
            return RegionMemoryValue::Forward(entry_reaching_def);
        }

        let (body_pred, reaching_body) = block_mem_val(
            self.loop_body(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );

        let phi_inputs = small_map! {
            MemoryRegionPredecessor::Parent => entry_reaching_def,
            body_pred => reaching_body
        };

        region_phis.insert(body_region, phi_inputs.clone());
        RegionMemoryValue::RegionPhi(phi_inputs)
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for RangeLoopOp {
    fn entry_successor_operands(&self, _ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        vec![]
    }

    fn successor_regions(&self, ctx: &Context, _pred: RegionPredecessor) -> Vec<RegionSuccessor> {
        // TODO: Loop interface for constant trip count
        vec![self.loop_region(ctx).into(), RegionSuccessor::AfterOp]
    }

    fn successor_inputs(&self, _ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        vec![]
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

#[op_interface_impl]
impl MemorySSARegionOpInterface for WhileOp {
    fn setup_memory_ssa(
        &self,
        ctx: &Context,
        state: &mut MemorySSAContext,
        reaching_def: MemoryValue,
        has_memory_defs: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, MemoryValue, 2>,
    ) {
        let before_region = self.before_region(ctx);
        let after_region = self.after_region(ctx);
        if !has_memory_defs {
            regions_to_process.insert(before_region, reaching_def);
            regions_to_process.insert(after_region, reaching_def);
            return;
        }

        let new_arg = state.new_value_in_block(self.before_block(ctx));
        regions_to_process.insert(before_region, new_arg);

        let new_arg = state.new_value_in_block(self.after_block(ctx));
        regions_to_process.insert(after_region, new_arg);
    }

    fn finalize_memory_ssa(
        &self,
        ctx: &Context,
        _state: &mut MemorySSAContext,
        entry_reaching_def: MemoryValue,
        has_memory_defs: bool,
        reaching_at_region_entry: &HMap<Ptr<Region>, MemoryValue>,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, MemoryValue>,
        region_phis: &mut SmallMap<Ptr<Region>, RegionMemoryPhiInputs, 2>,
    ) -> RegionMemoryValue {
        if !has_memory_defs {
            return RegionMemoryValue::Forward(entry_reaching_def);
        }

        let before_region = self.before_region(ctx);
        let after_region = self.after_region(ctx);

        let arg = reaching_at_region_entry[&before_region];
        let (before_pred, reaching_before) =
            block_mem_val(self.before_block(ctx), arg, reaching_at_block_end);

        let arg = reaching_at_region_entry[&after_region];
        let (after_pred, reaching_after) =
            block_mem_val(self.after_block(ctx), arg, reaching_at_block_end);

        let inputs_before = small_map! {
            MemoryRegionPredecessor::Parent => entry_reaching_def,
            after_pred => reaching_after
        };
        let inputs_after = small_map!(before_pred => reaching_before);

        region_phis.insert(before_region, inputs_before);
        region_phis.insert(after_region, inputs_after);

        RegionMemoryValue::Forward(reaching_before)
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for WhileOp {
    fn entry_successor_operands(&self, _ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        vec![]
    }

    fn successor_regions(&self, ctx: &Context, pred: RegionPredecessor) -> Vec<RegionSuccessor> {
        match pred {
            RegionPredecessor::Parent => vec![self.before_region(ctx).into()],
            RegionPredecessor::Terminator(term) => {
                let op = term.deref(ctx).get_operation();
                let parent = op.deref(ctx).get_parent_region(ctx).unwrap();
                if parent == self.after_region(ctx) {
                    vec![self.before_region(ctx).into()]
                } else {
                    vec![RegionSuccessor::AfterOp, self.after_region(ctx).into()]
                }
            }
        }
    }

    fn successor_inputs(&self, _ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        vec![]
    }
}
