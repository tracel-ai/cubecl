use std::cell::RefCell;

use cubecl_core::ir::{Branch, Metadata, Operation, Operator, UnaryOperator, Variable};
use stable_vec::StableVec;

use crate::{visit_noop, AtomicCounter, Optimizer};

use super::OptimizationPass;

pub struct InlineAssignments;

impl OptimizationPass for InlineAssignments {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        while search_loop(opt) {
            changes.inc();
        }
    }
}

fn search_loop(opt: &mut Optimizer) -> bool {
    for node in opt.program.node_indices().collect::<Vec<_>>() {
        let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

        for idx in ops {
            let op = opt.program[node].ops.borrow()[idx].clone();
            if let Operation::Operator(Operator::Assign(op)) = op {
                if op.input.is_immutable() && op.out.is_immutable() {
                    opt.visit_all(
                        |_, var| {
                            if *var == op.out {
                                *var = op.input
                            }
                        },
                        visit_noop,
                    );
                    opt.program[node].ops.borrow_mut().remove(idx);
                    return true;
                }
            }
        }
    }

    false
}

pub struct MergeSameExpressions;

impl OptimizationPass for MergeSameExpressions {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for node in opt.node_ids() {
            let ops = opt.program[node].ops.clone();
            let indices = ops.borrow().indices().collect::<Vec<_>>();
            for (i, idx) in indices.iter().enumerate() {
                check_op(opt, i, *idx, &ops, &indices, &changes);
            }
        }
    }
}

fn check_op(
    opt: &mut Optimizer,
    i: usize,
    idx: usize,
    ops: &RefCell<StableVec<Operation>>,
    indices: &[usize],
    changes: &AtomicCounter,
) -> Option<()> {
    let mut op = ops.borrow()[idx].clone();
    let out = get_out(opt, &mut op)?;
    let mut is_mut = false;
    opt.visit_operation(
        &mut op,
        |_, var| {
            if !var.is_immutable() {
                is_mut = true;
            }
        },
        visit_noop,
    );
    for rhs_idx in indices.iter().skip(i + 1) {
        if rhs_eq(&op, &ops.borrow()[*rhs_idx]) {
            let rhs_out = get_out(opt, &mut ops.borrow_mut()[*rhs_idx])?;
            ops.borrow_mut()[*rhs_idx] = Operator::Assign(UnaryOperator {
                input: out,
                out: rhs_out,
            })
            .into();
            changes.inc();
        }
    }
    Some(())
}

fn get_out(opt: &mut Optimizer, op: &mut Operation) -> Option<Variable> {
    let mut out = None;
    opt.visit_operation(op, visit_noop, |_, var| out = Some(*var));
    out
}

fn rhs_eq(lhs: &Operation, rhs: &Operation) -> bool {
    match (lhs, rhs) {
        (Operation::Operator(lhs), Operation::Operator(rhs)) => operator_rhs_eq(lhs, rhs),
        (Operation::Metadata(lhs), Operation::Metadata(rhs)) => metadata_rhs_eq(lhs, rhs),
        (Operation::Branch(lhs), Operation::Branch(rhs)) => branch_rhs_eq(lhs, rhs),
        _ => false,
    }
}

fn branch_rhs_eq(lhs: &Branch, rhs: &Branch) -> bool {
    match (lhs, rhs) {
        (Branch::Select(lhs), Branch::Select(rhs)) => {
            lhs.cond == rhs.cond && lhs.then == rhs.then && lhs.or_else == rhs.or_else
        }
        _ => false,
    }
}

fn operator_rhs_eq(lhs: &Operator, rhs: &Operator) -> bool {
    match (lhs, rhs) {
        (Operator::Add(lhs), Operator::Add(rhs))
        | (Operator::And(lhs), Operator::And(rhs))
        | (Operator::BitwiseAnd(lhs), Operator::BitwiseAnd(rhs))
        | (Operator::BitwiseOr(lhs), Operator::BitwiseOr(rhs))
        | (Operator::BitwiseXor(lhs), Operator::BitwiseXor(rhs))
        | (Operator::Div(lhs), Operator::Div(rhs))
        | (Operator::Dot(lhs), Operator::Dot(rhs))
        | (Operator::Equal(lhs), Operator::Equal(rhs))
        | (Operator::Greater(lhs), Operator::Greater(rhs))
        | (Operator::GreaterEqual(lhs), Operator::GreaterEqual(rhs))
        | (Operator::Index(lhs), Operator::Index(rhs))
        | (Operator::IndexAssign(lhs), Operator::IndexAssign(rhs))
        | (Operator::Lower(lhs), Operator::Lower(rhs))
        | (Operator::LowerEqual(lhs), Operator::LowerEqual(rhs))
        | (Operator::Max(lhs), Operator::Max(rhs))
        | (Operator::Min(lhs), Operator::Min(rhs))
        | (Operator::Modulo(lhs), Operator::Modulo(rhs))
        | (Operator::Mul(lhs), Operator::Mul(rhs))
        | (Operator::NotEqual(lhs), Operator::NotEqual(rhs))
        | (Operator::Or(lhs), Operator::Or(rhs))
        | (Operator::Powf(lhs), Operator::Powf(rhs))
        | (Operator::Remainder(lhs), Operator::Remainder(rhs))
        | (Operator::ShiftLeft(lhs), Operator::ShiftLeft(rhs))
        | (Operator::ShiftRight(lhs), Operator::ShiftRight(rhs))
        | (Operator::Sub(lhs), Operator::Sub(rhs))
        | (Operator::UncheckedIndex(lhs), Operator::UncheckedIndex(rhs))
        | (Operator::UncheckedIndexAssign(lhs), Operator::UncheckedIndexAssign(rhs)) => {
            lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs
        }

        (Operator::Abs(lhs), Operator::Abs(rhs))
        | (Operator::Bitcast(lhs), Operator::Bitcast(rhs))
        | (Operator::Ceil(lhs), Operator::Ceil(rhs))
        | (Operator::Cos(lhs), Operator::Cos(rhs))
        | (Operator::Erf(lhs), Operator::Erf(rhs))
        | (Operator::Exp(lhs), Operator::Exp(rhs))
        | (Operator::Floor(lhs), Operator::Floor(rhs))
        | (Operator::Log(lhs), Operator::Log(rhs))
        | (Operator::Log1p(lhs), Operator::Log1p(rhs))
        | (Operator::Magnitude(lhs), Operator::Magnitude(rhs))
        | (Operator::Neg(lhs), Operator::Neg(rhs))
        | (Operator::Normalize(lhs), Operator::Normalize(rhs))
        | (Operator::Not(lhs), Operator::Not(rhs))
        | (Operator::Recip(lhs), Operator::Recip(rhs))
        | (Operator::Round(lhs), Operator::Round(rhs))
        | (Operator::Sin(lhs), Operator::Sin(rhs))
        | (Operator::Sqrt(lhs), Operator::Sqrt(rhs))
        | (Operator::Tanh(lhs), Operator::Tanh(rhs)) => lhs.input == rhs.input,

        (Operator::Clamp(lhs), Operator::Clamp(rhs)) => {
            lhs.input == rhs.input
                && lhs.min_value == rhs.min_value
                && lhs.max_value == rhs.max_value
        }
        (Operator::Fma(lhs), Operator::Fma(rhs)) => {
            lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c
        }
        (Operator::InitLine(lhs), Operator::InitLine(rhs)) => lhs.inputs == rhs.inputs,
        (Operator::Slice(lhs), Operator::Slice(rhs)) => {
            lhs.input == rhs.input && lhs.start == rhs.start && lhs.end == rhs.end
        }
        _ => false,
    }
}

fn metadata_rhs_eq(lhs: &Metadata, rhs: &Metadata) -> bool {
    match (lhs, rhs) {
        (
            Metadata::Stride {
                dim: dim_lhs,
                var: var_lhs,
                ..
            },
            Metadata::Stride { dim, var, .. },
        ) => dim_lhs == dim && var_lhs == var,
        (
            Metadata::Shape {
                dim: dim_lhs,
                var: var_lhs,
                ..
            },
            Metadata::Shape { dim, var, .. },
        ) => dim_lhs == dim && var_lhs == var,
        (Metadata::Length { var: var_lhs, .. }, Metadata::Length { var, .. }) => var_lhs == var,
        _ => false,
    }
}
