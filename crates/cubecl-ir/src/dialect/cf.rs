//! Mainly for testing, but can be used as an intermediate CFG dialect that still uses `cube.bool`

use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::{attributes::IntegerAttr, op_interfaces, types::IntegerType},
    combine::{self, Parser, between, parser::char::spaces, token},
    common_traits::Named,
    identifier::Identifier,
    indented_block, input_err,
    irfmt::{
        self,
        parsers::{
            block_opd_parser, delimited_list_parser, process_parsed_ssa_defs, spaced,
            ssa_opd_parser,
        },
        printers::{iter_with_sep, list_with_sep},
    },
    location::Location,
    op::OpObj,
    opts::constants::BranchOpFoldInterface,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{Printable, indented_nl},
    verify_err,
};
use thiserror::Error;

use crate::{
    attributes::{BoolAttr, IntegerVecAttr, ZeroAttr},
    prelude::*,
    types::scalar::BoolType,
};

#[pliron_op(
    name = "cf.branch",
    format = "succ($0) `(` operands(CharSpace(`,`)) `)`",
    interfaces = [
        IsTerminatorInterface,
        NResultsInterface<0>,
        NSuccsInterface<1>,
        OneSuccInterface
    ],
    verifier = "succ"
)]
pub struct BranchOp;

#[op_interface_impl]
impl BranchOpInterface for BranchOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        self.get_operation().deref(ctx).operands().collect()
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        Operation::push_operand(self.get_operation(), ctx, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }
}

impl BranchOp {
    /// Create a new [`BranchOp`].
    pub fn new(ctx: &mut Context, dest: Ptr<BasicBlock>, dest_opds: Vec<Value>) -> Self {
        BranchOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                dest_opds,
                vec![dest],
                0,
            ),
        }
    }
}

#[pliron_op(
    name = "cf.branch_conditional",
    operands = (condition: BoolType),
    verifier = "succ"
)]
#[op_interfaces(IsTerminatorInterface, NResultsInterface<0>, NSuccsInterface<2>, OperandSegmentInterface)]
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

impl Printable for BranchConditionalOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let op = self.get_operation().deref(ctx);
        let condition = self.get_operand_condition(ctx);
        let true_dest_opds = self.successor_operands(ctx, 0);
        let false_dest_opds = self.successor_operands(ctx, 1);
        let res = write!(
            f,
            "{} if {} ^{}({}) else ^{}({})",
            Self::get_opid_static(),
            condition.print(ctx, state),
            op.get_successor(0).deref(ctx).unique_name(ctx),
            iter_with_sep(
                true_dest_opds.iter(),
                pliron::printable::ListSeparator::CharSpace(',')
            )
            .print(ctx, state),
            op.get_successor(1).deref(ctx).unique_name(ctx),
            iter_with_sep(
                false_dest_opds.iter(),
                pliron::printable::ListSeparator::CharSpace(',')
            )
            .print(ctx, state),
        );
        res
    }
}

impl Parsable for BranchConditionalOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }

        // Parse the condition operand.
        let r#if = irfmt::parsers::spaced::<StateStream, _>(combine::parser::char::string("if"));

        let condition = ssa_opd_parser();

        let true_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());

        let r_else =
            irfmt::parsers::spaced::<StateStream, _>(combine::parser::char::string("else"));

        let false_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());

        let final_parser = r#if
            .with(spaced(condition))
            .and(spaced(block_opd_parser()))
            .and(true_operands)
            .and(spaced(r_else).with(spaced(block_opd_parser()).and(false_operands)));

        final_parser
            .then(
                move |(((condition, true_dest), true_dest_opds), (false_dest, false_dest_opds))| {
                    let results = results.clone();
                    combine::parser(move |parsable_state: &mut StateStream<'a>| {
                        let ctx = &mut parsable_state.state.ctx;
                        let op = BranchConditionalOp::new(
                            ctx,
                            condition,
                            true_dest,
                            true_dest_opds.clone(),
                            false_dest,
                            false_dest_opds.clone(),
                        );

                        process_parsed_ssa_defs(parsable_state, &results, op.get_operation())?;
                        Ok(OpObj::new(op)).into_parse_result()
                    })
                },
            )
            .parse_stream(state_stream)
            .into()
    }
}

#[op_interface_impl]
impl BranchOpInterface for BranchConditionalOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(
            succ_idx == 0 || succ_idx == 1,
            "CondBrOp has exactly two successors"
        );

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

#[pliron_op(
    name = "cf.switch",
    operands = (value: IntegerType),
    attributes = (switch_case_values: IntegerVecAttr)
)]
#[op_interfaces(IsTerminatorInterface, NResultsInterface<0>, OperandSegmentInterface)]
pub struct SwitchOp;

/// One case of a switch statement.
#[derive(Clone)]
pub struct SwitchCase {
    /// The value being matched against.
    pub value: IntegerAttr,
    /// The destination block to jump to if this case is taken.
    pub dest: Ptr<BasicBlock>,
    /// The operands to pass to the destination block.
    pub dest_opds: Vec<Value>,
}

impl Printable for SwitchCase {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "{{ {}: ^{}({}) }}",
            self.value.print(ctx, state),
            self.dest.deref(ctx).unique_name(ctx),
            list_with_sep(
                &self.dest_opds,
                pliron::printable::ListSeparator::CharSpace(',')
            )
            .print(ctx, state)
        )
    }
}

impl Parsable for SwitchCase {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            token('{'),
            token('}'),
            (
                spaced(IntegerAttr::parser(())),
                spaced(token(':')),
                spaced(block_opd_parser()),
                delimited_list_parser('(', ')', ',', ssa_opd_parser()),
                spaces(),
            ),
        );

        let ((value, _colon, dest, dest_opds, _spaces), _) =
            parser.parse_stream(state_stream).into_result()?;

        Ok(SwitchCase {
            value,
            dest,
            dest_opds,
        })
        .into_parse_result()
    }
}

impl Printable for SwitchOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let op = self.get_operation().deref(ctx);
        let value = self.get_operand_value(ctx);

        let default_successor = op
            .successors()
            .next()
            .expect("SwitchOp must have at least one successor");
        let num_total_successors = op.get_num_successors();

        write!(
            f,
            "{} {}, ^{}({})",
            Self::get_opid_static(),
            value.print(ctx, state),
            default_successor.unique_name(ctx).print(ctx, state),
            iter_with_sep(
                self.successor_operands(ctx, 0).iter(),
                pliron::printable::ListSeparator::CharSpace(',')
            )
            .print(ctx, state),
        )?;

        if num_total_successors < 2 {
            writeln!(f, "[]")?;
            return Ok(());
        }

        let cases = self.cases(ctx);

        write!(f, "{}[", indented_nl(state))?;
        indented_block!(state, {
            write!(f, "{}", indented_nl(state))?;
            list_with_sep(&cases, pliron::printable::ListSeparator::CharNewline(','))
                .fmt(ctx, state, f)?;
        });
        write!(f, "{}]", indented_nl(state))?;

        Ok(())
    }
}

impl Parsable for SwitchOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !arg.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, arg.len())
            )?
        }

        // Parse the condition operand.
        let condition = ssa_opd_parser().skip(spaced(token(',')));
        let default_successor = block_opd_parser();
        let default_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());
        let cases = delimited_list_parser('[', ']', ',', SwitchCase::parser(()));

        let final_parser = spaced(condition)
            .and(default_successor)
            .skip(spaces())
            .and(default_operands)
            .skip(spaces())
            .and(cases);

        final_parser
            .then(
                move |(((condition, default_dest), default_dest_opds), cases)| {
                    let results = arg.clone();
                    combine::parser(move |parsable_state: &mut StateStream<'a>| {
                        let ctx = &mut parsable_state.state.ctx;
                        let op = SwitchOp::new(
                            ctx,
                            condition,
                            default_dest,
                            default_dest_opds.clone(),
                            cases.clone(),
                        );

                        process_parsed_ssa_defs(parsable_state, &results, op.get_operation())?;
                        Ok(OpObj::new(op)).into_parse_result()
                    })
                },
            )
            .parse_stream(state_stream)
            .into()
    }
}

impl SwitchOp {
    /// Create a new [`SwitchOp`].
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        default_dest: Ptr<BasicBlock>,
        default_dest_opds: Vec<Value>,
        cases: Vec<SwitchCase>,
    ) -> Self {
        let case_values: Vec<IntegerAttr> = cases.iter().map(|case| case.value.clone()).collect();

        let case_operands = cases
            .iter()
            .map(|case| case.dest_opds.clone())
            .collect::<Vec<_>>();

        let mut operand_segments = vec![vec![condition], default_dest_opds];
        operand_segments.extend(case_operands);
        let (operands, segment_sizes) = Self::compute_segment_sizes(operand_segments);

        let case_dests = cases.iter().map(|case| case.dest);
        let successors = vec![default_dest].into_iter().chain(case_dests).collect();
        let op = SwitchOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                successors,
                0,
            ),
        };

        // Set the operand segment sizes attribute.
        op.set_operand_segment_sizes(ctx, segment_sizes);
        // Set the case values
        op.set_attr_switch_case_values(ctx, IntegerVecAttr(case_values));
        op
    }

    /// Get the cases of this switch operation.
    /// (The default case cannot be / isn't included here).
    pub fn cases(&self, ctx: &Context) -> Vec<SwitchCase> {
        let case_values = &*self
            .get_attr_switch_case_values(ctx)
            .expect("SwitchOp missing or incorrect case values attribute");

        let op = self.get_operation().deref(ctx);
        // Skip the first one, which is the default successor.
        let successors = op.successors().skip(1);

        successors
            .zip(case_values.0.iter())
            .enumerate()
            .map(|(i, (dest, value))| {
                // i+1 here because the first successor is the default destination.
                let dest_opds = self.successor_operands(ctx, i + 1);
                SwitchCase {
                    value: value.clone(),
                    dest,
                    dest_opds,
                }
            })
            .collect()
    }

    /// Get the default destination of this switch operation.
    pub fn default_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }

    /// Get the operands to pass to the default destination.
    pub fn default_dest_operands(&self, ctx: &Context) -> Vec<Value> {
        self.successor_operands(ctx, 0)
    }
}

#[op_interface_impl]
impl BranchOpInterface for SwitchOp {
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

#[derive(Error, Debug)]
pub enum SwitchOpVerifyErr {
    #[error("SwitchOp has no or incorrect case values attribute")]
    CaseValuesAttrErr,
    #[error("SwitchOp has no or incorrect default destination")]
    DefaultDestErr,
}

impl Verify for SwitchOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        let op = &*self.get_operation().deref(ctx);
        if op.get_num_successors() < 1 {
            verify_err!(loc.clone(), SwitchOpVerifyErr::DefaultDestErr)?;
        }

        Ok(())
    }
}

#[op_interface_impl]
impl BranchOpFoldInterface for BranchOp {
    fn check_fold(&self, ctx: &Context, _operands: &[Option<AttrObj>]) -> Vec<Ptr<BasicBlock>> {
        self.get_operation().deref(ctx).successors().collect()
    }
    fn fold_in_place(
        &self,
        _ctx: &mut Context,
        _ops: &[Option<AttrObj>],
        _rw: &mut dyn Rewriter,
    ) -> IRStatus {
        IRStatus::Unchanged
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
        std::println!("cond: {}", cond_attr.disp(ctx));
        let zero = cond_attr.downcast_ref::<ZeroAttr>().map(|_| false);
        let bool = cond_attr.downcast_ref::<BoolAttr>().map(|it| it.0);
        let Some(const_cond) = zero.or(bool) else {
            let num_successors = self.get_operation().deref(ctx).successors().count();
            return (0..num_successors).collect();
        };
        let taken = if const_cond { 0 } else { 1 };
        std::println!("taken: {taken}");
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
        _ctx: &mut Context,
        _ops: &[Option<AttrObj>],
        _rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        IRStatus::Unchanged
    }
}

#[op_interface_impl]
impl BranchOpFoldInterface for SwitchOp {
    fn check_fold(&self, ctx: &Context, operands: &[Option<AttrObj>]) -> Vec<Ptr<BasicBlock>> {
        let successors: Vec<Ptr<BasicBlock>> =
            self.get_operation().deref(ctx).successors().collect();
        let Some(cond_attr) = operands.first().and_then(|o| o.as_ref()) else {
            return successors;
        };
        let cond_int = cond_attr
            .downcast_ref::<IntegerAttr>()
            .expect("Switch condition operand must be an IntegerAttr")
            .value();
        // Successor 0 is the default destination; successors 1..N correspond to case_values[0..N-1].
        let case_values = self
            .get_attr_switch_case_values(ctx)
            .expect("SwitchOp missing case values attribute");
        let taken = case_values
            .0
            .iter()
            .position(|case| case.value() == cond_int)
            .map(|i| i + 1)
            .unwrap_or(0);
        vec![successors[taken]]
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
