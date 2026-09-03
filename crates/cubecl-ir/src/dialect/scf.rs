//! Lower level branch dialect for targets that support block args/phi. Separate from the base
//! dialect because lifting back out of this is too hard for C++/WGSL. This can be used for
//! optimizations that require structured control flow but can also support transformations like
//! `mem2reg` or PRE that require threading SSA values in and out of the control flow ops.
//!
//! All terminators are reused from `branch`.

use cubecl_macros_internal::NamedRewrite;
use itertools::Itertools;
use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::attributes::IntegerAttr,
    combine::{
        Parser, optional,
        parser::char::{self, spaces},
    },
    identifier::Identifier,
    input_err,
    irbuild::inserter::OpInsertionPoint,
    irfmt::parsers::{delimited_list_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser},
    linked_list::ContainsLinkedList,
    location::Location,
    op::{OpBox, OpObj},
    opts::{dce::SideEffects, mem2reg::AllocInfo},
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::Printable,
    region::Region,
    utils::table::{HMap, SmallMap},
};

use crate::{
    attributes::{BoolAttr, IntegerVecAttr, ZeroAttr},
    dialect::{
        BlockPtrExt,
        branch::{self, ConditionOp, YieldOp, block_side_effects},
    },
    interfaces::{
        CanonicalizeInterface,
        control_flow::{
            InvocationBounds, RegionBranchOpInterface, RegionPredecessor, RegionSuccessor,
        },
        memory_slot::PromotableRegionOpInterface,
    },
    prelude::*,
    types::scalar::BoolType,
};

#[pliron_op(
    name = "scf.if",
    format = "$0 ` : ` types(CharSpace(`,`)) ` then ` region($0) ` else ` region($1)",
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

    pub fn get_result(&self, ctx: &Context, res_idx: usize) -> Value {
        self.get_operation().deref(ctx).get_result(res_idx)
    }

    pub fn results(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().results(ctx)
    }

    pub fn result_types(&self, ctx: &Context) -> Vec<TypeHandle> {
        self.get_operation().result_types(ctx)
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

#[op_interface_impl]
impl PromotableRegionOpInterface for IfOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        _: &AllocInfo,
        reaching_def: Value,
        _: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        regions_to_process.insert(self.then_region(ctx), reaching_def);
        regions_to_process.insert(self.else_region(ctx), reaching_def);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        entry_reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return entry_reaching_def;
        }

        update_terminator::<YieldOp>(
            ctx,
            self.then_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );
        update_terminator::<YieldOp>(
            ctx,
            self.else_block(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );

        let res_idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_result(ctx, res_idx)
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

    fn successor_inputs(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value> {
        match successor {
            RegionSuccessor::Region(_) => vec![],
            RegionSuccessor::AfterOp => self.results(ctx),
        }
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
    fn remove_unused_carried(&self, ctx: &mut Context) -> Result<()> {
        let results = self.get_operation().results(ctx);
        for (idx, res) in results.into_iter().enumerate().rev() {
            if !res.is_used(ctx) {
                remove_from_terminator::<YieldOp>(ctx, self.then_block(ctx), idx);
                remove_from_terminator::<YieldOp>(ctx, self.else_block(ctx), idx);
                Operation::remove_result(self.get_operation(), ctx, idx);
            }
        }
        Ok(())
    }

    fn fold(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let op = self.get_operation();
        let operands = const_operands(ctx, op);
        let valid_branches = self.entry_successor_regions(ctx, &operands);
        let &[RegionSuccessor::Region(taken)] = valid_branches.as_slice() else {
            return Ok(());
        };
        let taken = taken.deref(ctx).get_entry_block().unwrap();

        let term = taken.deref(ctx).get_terminator(ctx);
        if let Some(term) = term
            && term.is_op::<YieldOp>(ctx)
        {
            let results = self.results(ctx);
            let yielded = term.operands(ctx);
            assert_eq!(results.len(), yielded.len(), "Yield doesn't match results");
            for (res, yielded) in results.into_iter().zip(yielded) {
                rewriter.replace_value_uses_with(ctx, res, yielded);
                Operation::pop_operand(term, ctx);
            }
        }

        inline_block(ctx, rewriter, taken, OpInsertionPoint::BeforeOperation(op));
        rewriter.erase_operation(ctx, op);

        Ok(())
    }
}

#[op_interface_impl]
impl CanonicalizeInterface for IfOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        self.remove_unused_carried(ctx)?;
        self.fold(ctx, rewriter)?;
        Ok(())
    }
}

#[pliron_op(
    name = "scf.switch",
    format,
    attributes = (scf_switch_cases: IntegerVecAttr),
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

    pub fn case_regions(&self, ctx: &Context) -> Vec<Ptr<Region>> {
        self.get_operation().deref(ctx).regions().skip(1).collect()
    }

    pub fn case_blocks(&self, ctx: &Context) -> Vec<Ptr<BasicBlock>> {
        self.case_regions(ctx)
            .iter()
            .map(|reg| reg.deref(ctx).get_entry_block().unwrap())
            .collect()
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
            let value = cases[i].clone();
            let block = self.get_body(ctx, i + 1);
            (value, block)
        });
        out.collect()
    }

    pub fn cases_regions(&self, ctx: &Context) -> Vec<(IntegerAttr, Ptr<Region>)> {
        let cases = self.get_attr_scf_switch_cases(ctx).unwrap().clone().0;
        let out = (0..cases.len()).map(|i| {
            let value = cases[i].clone();
            let block = self.get_operation().deref(ctx).get_region(i + 1);
            (value, block)
        });
        out.collect()
    }

    pub fn cases_values(&self, ctx: &Context) -> Vec<IntegerAttr> {
        self.get_attr_scf_switch_cases(ctx).unwrap().0.clone()
    }

    pub fn get_case_destinations(&self, ctx: &Context) -> Vec<Ptr<BasicBlock>> {
        let op = self.get_operation().deref(ctx);
        (1..op.regions().count())
            .map(|i| self.get_body(ctx, i))
            .collect()
    }

    pub fn set_attr_cases(&self, ctx: &Context, cases: impl IntoIterator<Item = IntegerAttr>) {
        self.set_attr_scf_switch_cases(ctx, IntegerVecAttr(cases.into_iter().collect()));
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

#[op_interface_impl]
impl PromotableRegionOpInterface for SwitchOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        _: &AllocInfo,
        reaching_def: Value,
        _: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        regions_to_process.insert(self.default_region(ctx), reaching_def);
        for case_region in self.case_regions(ctx) {
            regions_to_process.insert(case_region, reaching_def);
        }
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return reaching_def;
        }

        update_terminator::<YieldOp>(
            ctx,
            self.default_block(ctx),
            reaching_def,
            reaching_at_block_end,
        );
        for case_block in self.case_blocks(ctx) {
            update_terminator::<YieldOp>(ctx, case_block, reaching_def, reaching_at_block_end);
        }

        let res_idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(res_idx)
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

    fn successor_inputs(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value> {
        match successor {
            RegionSuccessor::Region(_) => vec![],
            RegionSuccessor::AfterOp => self.results(ctx),
        }
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
    fn remove_unused_carried(&self, ctx: &mut Context) -> Result<()> {
        let results = self.get_operation().results(ctx);
        for (idx, res) in results.into_iter().enumerate().rev() {
            if !res.is_used(ctx) {
                remove_from_terminator::<YieldOp>(ctx, self.default_block(ctx), idx);
                for case_block in self.case_blocks(ctx) {
                    remove_from_terminator::<YieldOp>(ctx, case_block, idx);
                }
                Operation::remove_result(self.get_operation(), ctx, idx);
            }
        }
        Ok(())
    }

    fn fold(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        let op = self.get_operation();
        let operands = const_operands(ctx, op);
        let valid_branches = self.entry_successor_regions(ctx, &operands);
        let &[RegionSuccessor::Region(taken)] = valid_branches.as_slice() else {
            return Ok(());
        };
        let taken = taken.deref(ctx).get_entry_block().unwrap();

        let term = taken.deref(ctx).get_terminator(ctx);
        if let Some(term) = term
            && term.is_op::<YieldOp>(ctx)
        {
            let results = self.results(ctx);
            let yielded = term.operands(ctx);
            assert_eq!(results.len(), yielded.len(), "Yield doesn't match results");
            for (res, yielded) in results.into_iter().zip(yielded) {
                rewriter.replace_value_uses_with(ctx, res, yielded);
                Operation::pop_operand(term, ctx);
            }
        }

        inline_block(ctx, rewriter, taken, OpInsertionPoint::BeforeOperation(op));
        rewriter.erase_operation(ctx, op);

        Ok(())
    }
}

#[op_interface_impl]
impl CanonicalizeInterface for SwitchOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut MatchRewriter) -> Result<()> {
        self.remove_unused_carried(ctx)?;
        self.fold(ctx, rewriter)?;
        Ok(())
    }
}

#[pliron_op(name = "scf.for", verifier = "succ")]
#[op_interfaces(
    OneRegionInterface,
    SingleBlockRegionInterface,
    OperandSegmentInterface
)]
pub struct RangeLoopOp;

impl RangeLoopOp {
    pub fn new(
        ctx: &mut Context,
        results: Vec<TypeHandle>,
        start: Value,
        end: Value,
        step: Value,
        carried_values_init: Vec<Value>,
    ) -> Self {
        let iter_ty = start.get_type(ctx);
        let mut body_args = vec![iter_ty];
        body_args.extend(carried_values_init.iter().map(|it| it.get_type(ctx)));

        let (operands, segments) =
            Self::compute_segment_sizes(vec![vec![start, end, step], carried_values_init]);

        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                1,
            ),
        };

        op.set_operand_segment_sizes(ctx, segments);

        let body_region = op.loop_region(ctx);
        let body = BasicBlock::new(ctx, Some("body".try_into().unwrap()), body_args);
        body.insert_at_front(body_region, ctx);

        op
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

    pub fn initial_carried_values(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }

    pub fn push_initial_carried_value(&self, ctx: &mut Context, value: Value) -> usize {
        self.push_to_segment(ctx, 1, value)
    }

    pub fn remove_initial_carried_value(&self, ctx: &mut Context, opd_idx: usize) -> Value {
        self.remove_from_segment(ctx, 1, opd_idx)
    }

    pub fn carried_values(&self, ctx: &Context) -> Vec<Value> {
        self.loop_body(ctx).deref(ctx).arguments().skip(1).collect()
    }

    pub fn get_carried_value(&self, ctx: &Context, arg_idx: usize) -> Value {
        self.loop_body(ctx).deref(ctx).get_argument(arg_idx + 1)
    }

    pub fn push_carried_value(&self, ctx: &mut Context, ty: TypeHandle) -> usize {
        BasicBlock::push_argument(self.loop_body(ctx), ctx, ty)
    }

    pub fn remove_carried_value(&self, ctx: &Context, arg_idx: usize) {
        BasicBlock::remove_argument(self.loop_body(ctx), ctx, arg_idx + 1)
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

impl Printable for RangeLoopOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let results = self.results(ctx);
        if !results.is_empty() {
            let results = results.iter().map(|it| it.disp(ctx)).join(", ");
            write!(f, "{results} = ")?;
        }
        let args = self.initial_carried_values(ctx);
        write!(
            f,
            "{} {} to {} step {} ",
            self.get_opid(),
            self.start(ctx).disp(ctx),
            self.end(ctx).disp(ctx),
            self.step(ctx).disp(ctx),
        )?;

        if !args.is_empty() {
            let args = args.iter().map(|it| it.disp(ctx)).join(", ");
            write!(f, "iter_args({args})")?;
        }

        self.loop_region(ctx).fmt(ctx, state, f)
    }
}

impl Parsable for RangeLoopOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let cur_loc = state_stream.loc();
        let iter_args = delimited_list_parser('(', ')', ',', ssa_opd_parser());
        let mut iter_args_parser =
            optional(spaced(char::string("iter_args").with(iter_args))).skip(spaces());
        let mut range_parser = (
            ssa_opd_parser().skip(spaced(char::string("to"))),
            ssa_opd_parser().skip(spaced(char::string("step"))),
            ssa_opd_parser(),
        );

        let (start, end, step) = range_parser.parse_stream(state_stream).into_result()?.0;
        let args = iter_args_parser.parse_stream(state_stream).into_result()?.0;
        let args = args.unwrap_or_default();

        if args.len() != results.len() {
            input_err!(
                cur_loc,
                "expected {} results based on iter vars, got {} during parsing",
                args.len(),
                results.len()
            )?;
        }

        let result_types = args
            .iter()
            .map(|it| it.get_type(state_stream.state.ctx))
            .collect();
        let (operands, segments) = Self::compute_segment_sizes(vec![vec![start, end, step], args]);

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            0,
        );
        process_parsed_ssa_defs(state_stream, &results, op)?;

        Region::parse(state_stream, op)?;
        let op = RangeLoopOp { op };
        op.set_operand_segment_sizes(state_stream.state.ctx, segments);

        Ok(OpBox::new(op)).into_parse_result()
    }
}

#[op_interface_impl]
impl BranchToSCFOp for branch::RangeLoopOp {
    fn to_scf(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let opds = self.get_operation().operands(ctx);
        let (operands, segments) = RangeLoopOp::compute_segment_sizes(vec![opds, vec![]]);
        let info = RangeLoopOp::get_concrete_op_info();
        let op = Operation::new(ctx, info, vec![], operands, vec![], 0);

        let regions = self.get_operation().regions(ctx);
        for region in regions {
            Region::move_to_op(region, op, ctx);
        }

        rewriter.append_operation(ctx, op);
        rewriter.replace_operation(ctx, self.get_operation(), op);

        let op = RangeLoopOp { op };
        op.set_operand_segment_sizes(ctx, segments);

        Ok(())
    }
}

#[op_interface_impl]
impl PromotableRegionOpInterface for RangeLoopOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        let body_region = self.loop_region(ctx);
        if !has_value_stores {
            regions_to_process.insert(body_region, reaching_def);
            return;
        }

        self.push_initial_carried_value(ctx, reaching_def);
        let idx = BasicBlock::push_argument(self.loop_body(ctx), ctx, alloc.ty);
        let new_arg = self.loop_body(ctx).deref(ctx).get_argument(idx);
        regions_to_process.insert(body_region, new_arg);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        entry_reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return entry_reaching_def;
        }

        update_terminator::<YieldOp>(
            ctx,
            self.loop_body(ctx),
            entry_reaching_def,
            reaching_at_block_end,
        );

        let idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(idx)
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for RangeLoopOp {
    fn entry_successor_operands(&self, ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        self.initial_carried_values(ctx)
    }

    fn successor_regions(&self, ctx: &Context, _pred: RegionPredecessor) -> Vec<RegionSuccessor> {
        // TODO: Loop interface for constant trip count
        vec![self.loop_region(ctx).into(), RegionSuccessor::AfterOp]
    }

    fn successor_inputs(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value> {
        match successor {
            RegionSuccessor::Region(_) => self.carried_values(ctx),
            RegionSuccessor::AfterOp => self.results(ctx),
        }
    }
}

#[op_interface_impl]
impl CanonicalizeInterface for RangeLoopOp {
    fn canonicalize(&self, ctx: &mut Context, _rewriter: &mut MatchRewriter) -> Result<()> {
        let results = self.get_operation().results(ctx);
        let body_block = self.loop_body(ctx);
        for (idx, res) in results.into_iter().enumerate().rev() {
            let carried = self.get_carried_value(ctx, idx);
            if !res.is_used(ctx) && only_used_for_forward::<YieldOp>(ctx, body_block, idx, carried)
            {
                remove_from_terminator::<YieldOp>(ctx, body_block, idx);
                self.remove_initial_carried_value(ctx, idx);
                self.remove_carried_value(ctx, idx);
                Operation::remove_result(self.get_operation(), ctx, idx);
            }
        }
        Ok(())
    }
}

#[pliron_op(
    name = "scf.while",
    format = "operands(CharSpace(`,`)) ` : ` types(CharSpace(`,`)) region($0) ` do ` region($1)",
    verifier = "succ"
)]
#[op_interfaces(NRegionsInterface<2>, SingleBlockRegionInterface)]
pub struct WhileOp;

impl WhileOp {
    pub fn new(
        ctx: &mut Context,
        results: Vec<TypeHandle>,
        carried_values_init: Vec<Value>,
    ) -> Self {
        let carried_types = carried_values_init
            .iter()
            .map(|it| it.get_type(ctx))
            .collect::<Vec<_>>();

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            carried_values_init,
            vec![],
            2,
        );

        let before_region = op.deref_mut(ctx).get_region(0);
        let before_block = BasicBlock::new(
            ctx,
            Some("before".try_into().unwrap()),
            carried_types.clone(),
        );
        before_block.insert_at_front(before_region, ctx);

        let after_region = op.deref_mut(ctx).get_region(1);
        let after_block = BasicBlock::new(ctx, Some("after".try_into().unwrap()), carried_types);
        after_block.insert_at_front(after_region, ctx);

        Self { op }
    }

    pub fn initial_carried_values(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().operands(ctx)
    }

    pub fn push_initial_carried_value(&self, ctx: &mut Context, value: Value) -> usize {
        Operation::push_operand(self.get_operation(), ctx, value)
    }

    pub fn remove_initial_carried_value(&self, ctx: &mut Context, opd_idx: usize) -> Value {
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }

    pub fn get_before_carried_value(&self, ctx: &Context, arg_idx: usize) -> Value {
        self.before_block(ctx).deref(ctx).get_argument(arg_idx)
    }

    pub fn push_before_carried_value(&self, ctx: &mut Context, ty: TypeHandle) -> usize {
        BasicBlock::push_argument(self.before_block(ctx), ctx, ty)
    }

    pub fn remove_before_carried_value(&self, ctx: &mut Context, arg_idx: usize) {
        BasicBlock::remove_argument(self.before_block(ctx), ctx, arg_idx)
    }

    pub fn get_after_carried_value(&self, ctx: &Context, arg_idx: usize) -> Value {
        self.after_block(ctx).deref(ctx).get_argument(arg_idx)
    }

    pub fn push_after_carried_value(&self, ctx: &mut Context, ty: TypeHandle) -> usize {
        BasicBlock::push_argument(self.after_block(ctx), ctx, ty)
    }

    pub fn remove_after_carried_value(&self, ctx: &mut Context, arg_idx: usize) {
        BasicBlock::remove_argument(self.after_block(ctx), ctx, arg_idx)
    }

    pub fn before_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(0)
    }

    pub fn before_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_body(ctx, 0)
    }

    pub fn after_region(&self, ctx: &Context) -> Ptr<Region> {
        self.get_operation().deref(ctx).get_region(1)
    }

    pub fn after_block(&self, ctx: &Context) -> Ptr<BasicBlock> {
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
impl BranchToSCFOp for branch::WhileOp {
    fn to_scf(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        let opds = self.get_operation().operands(ctx);
        let info = WhileOp::get_concrete_op_info();
        let op = Operation::new(ctx, info, vec![], opds, vec![], 0);

        let regions = self.get_operation().regions(ctx);
        for region in regions {
            Region::move_to_op(region, op, ctx);
        }

        rewriter.append_operation(ctx, op);
        rewriter.replace_operation(ctx, self.get_operation(), op);

        Ok(())
    }
}

#[op_interface_impl]
impl PromotableRegionOpInterface for WhileOp {
    fn is_region_promotable(&self, _: &Context, _: &AllocInfo, _: Ptr<Region>, _: bool) -> bool {
        true
    }

    fn setup_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        regions_to_process: &mut SmallMap<Ptr<Region>, Value, 2>,
    ) {
        let before_region = self.before_region(ctx);
        let after_region = self.after_region(ctx);
        if !has_value_stores {
            regions_to_process.insert(before_region, reaching_def);
            regions_to_process.insert(after_region, reaching_def);
            return;
        }

        self.push_initial_carried_value(ctx, reaching_def);

        let idx = BasicBlock::push_argument(self.before_block(ctx), ctx, alloc.ty);
        let new_arg = self.before_block(ctx).deref(ctx).get_argument(idx);
        regions_to_process.insert(before_region, new_arg);

        let idx = BasicBlock::push_argument(self.after_block(ctx), ctx, alloc.ty);
        let new_arg = self.after_block(ctx).deref(ctx).get_argument(idx);
        regions_to_process.insert(after_region, new_arg);
    }

    fn finalize_promotion(
        &self,
        ctx: &mut Context,
        alloc: &AllocInfo,
        reaching_def: Value,
        has_value_stores: bool,
        reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
    ) -> Value {
        if !has_value_stores {
            return reaching_def;
        }

        let before = self.before_block(ctx);
        let last_arg = before.deref(ctx).arguments().last();
        update_terminator::<ConditionOp>(ctx, before, last_arg.unwrap(), reaching_at_block_end);

        let after = self.after_block(ctx);
        let last_arg = after.deref(ctx).arguments().last();
        update_terminator::<YieldOp>(ctx, after, last_arg.unwrap(), reaching_at_block_end);

        let idx = Operation::push_result(self.get_operation(), ctx, alloc.ty);
        self.get_operation().deref(ctx).get_result(idx)
    }
}

#[op_interface_impl]
impl RegionBranchOpInterface for WhileOp {
    fn entry_successor_operands(&self, ctx: &Context, _successor: RegionSuccessor) -> Vec<Value> {
        self.initial_carried_values(ctx)
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

    fn successor_inputs(&self, ctx: &Context, successor: RegionSuccessor) -> Vec<Value> {
        match successor {
            RegionSuccessor::Region(region) if region == self.before_region(ctx) => {
                self.before_block(ctx).arguments(ctx)
            }
            RegionSuccessor::Region(_) => self.after_block(ctx).arguments(ctx),
            RegionSuccessor::AfterOp => self.results(ctx),
        }
    }
}

#[op_interface_impl]
impl CanonicalizeInterface for WhileOp {
    fn canonicalize(&self, ctx: &mut Context, _rewriter: &mut MatchRewriter) -> Result<()> {
        let results = self.get_operation().results(ctx);
        let before_block = self.before_block(ctx);
        let after_block = self.after_block(ctx);
        for (idx, res) in results.into_iter().enumerate().rev() {
            let before_carried = self.get_before_carried_value(ctx, idx);
            let after_carried = self.get_after_carried_value(ctx, idx);
            if !res.is_used(ctx)
                && only_used_for_forward::<ConditionOp>(ctx, before_block, idx + 1, before_carried)
                && only_used_for_forward::<YieldOp>(ctx, after_block, idx, after_carried)
            {
                remove_from_terminator::<ConditionOp>(ctx, before_block, idx + 1);
                remove_from_terminator::<YieldOp>(ctx, after_block, idx);
                self.remove_initial_carried_value(ctx, idx);
                self.remove_before_carried_value(ctx, idx);
                self.remove_after_carried_value(ctx, idx);
                Operation::remove_result(self.get_operation(), ctx, idx);
            }
        }
        Ok(())
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

#[derive(Default, NamedRewrite)]
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

fn update_terminator<T: Op>(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    default_reaching_def: Value,
    reaching_at_block_end: &HMap<Ptr<BasicBlock>, Value>,
) {
    let Some(term) = block.deref(ctx).get_terminator(ctx) else {
        return;
    };
    if !term.is_op::<T>(ctx) {
        return;
    }
    let block_reaching_def = reaching_at_block_end.get(&block).copied();
    let block_reaching_def = block_reaching_def.unwrap_or(default_reaching_def);
    Operation::push_operand(term, ctx, block_reaching_def);
}

fn remove_from_terminator<T: Op>(ctx: &Context, block: Ptr<BasicBlock>, idx: usize) {
    let Some(term) = block.deref(ctx).get_terminator(ctx) else {
        return;
    };
    if !term.is_op::<T>(ctx) {
        return;
    }
    Operation::remove_operand(term, ctx, idx);
}

/// Detects circularly used loop args, where the arg is used only to be forwarded to the after block
/// or next iteration.
fn only_used_for_forward<T: Op>(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    idx: usize,
    val: Value,
) -> bool {
    if !val.is_used(ctx) {
        return true;
    }
    let Some(term) = block.deref(ctx).get_terminator(ctx) else {
        return false;
    };
    if !term.is_op::<T>(ctx) {
        return false;
    }
    let uses = val.uses(ctx);
    if uses.len() > 1 {
        return false;
    }
    uses[0].user_op() == term && uses[0].find_index(ctx) == idx
}
