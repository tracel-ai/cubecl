use std::{cell::RefCell, mem::take};

use cubecl_ir::{CoopMma, Instruction, Item, Operation, Operator, UnaryOperator};
use stable_vec::StableVec;

use crate::{AtomicCounter, Optimizer, visit_noop};

use super::OptimizerPass;

/// Inline constants or simple reassignments that don't change the type. This simplifies the code
/// and makes it easier to find optimizable expressions.
pub struct InlineAssignments;

impl OptimizerPass for InlineAssignments {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        while search_loop(opt) {
            changes.inc();
        }
    }
}

fn search_loop(opt: &mut Optimizer) -> bool {
    for node in opt.program.node_indices().collect::<Vec<_>>() {
        let mut removed_phi = Vec::new();
        // Remove trivial phi nodes left from PRE
        opt.program[node].phi_nodes.borrow_mut().retain(|it| {
            let reference = it.entries[0].value;
            if it.entries.iter().all(|it| it.value == reference) {
                removed_phi.push(it.clone());
                false
            } else {
                true
            }
        });

        if !removed_phi.is_empty() {
            let mut phi_assigns = removed_phi
                .into_iter()
                .map(|phi| Instruction::new(Operation::Copy(phi.entries[0].value), phi.out))
                .collect::<StableVec<_>>();

            let ops = take(&mut *opt.program[node].ops.borrow_mut());
            phi_assigns.extend(ops.into_iter().map(|(_, v)| v));
            *opt.program[node].ops.borrow_mut() = phi_assigns;
            return true;
        }

        let ops = opt.program[node].ops.borrow().indices().collect::<Vec<_>>();

        for idx in ops {
            let op = opt.program[node].ops.borrow()[idx].clone();
            match op.operation {
                Operation::Copy(input)
                | Operation::Operator(Operator::Cast(UnaryOperator { input }))
                | Operation::Operator(Operator::Reinterpret(UnaryOperator { input }))
                | Operation::CoopMma(CoopMma::Cast { input }) => {
                    if (input.is_immutable() || input.is_array())
                        && (op.out().is_immutable() || op.out().is_array())
                        && item_compatible(input.item, op.item())
                    {
                        opt.visit_all(
                            |_, var| {
                                if *var == op.out() {
                                    *var = input
                                }
                            },
                            visit_noop,
                        );
                        opt.program[node].ops.borrow_mut().remove(idx);
                        return true;
                    }
                }
                _ => {}
            }
        }
    }

    false
}

pub fn item_compatible(lhs: Item, rhs: Item) -> bool {
    let vectorization_lhs = lhs.vectorization.map(|it| it.get()).unwrap_or(1);
    let vectorization_rhs = rhs.vectorization.map(|it| it.get()).unwrap_or(1);
    vectorization_lhs == vectorization_rhs && lhs.elem() == rhs.elem()
}

/// Merge identical and immutable expressions in the same block into a single variable.
///
/// # Example
/// ```ignore
/// let a = rank - 2;
/// let b = rank - 1;
/// // in some other function
/// let x = rank - 2;
/// ```
/// would simplify to
/// ```ignore
/// let a = rank - 2;
/// let b = rank - 1;
/// let x = a;
/// ```
/// which can get further inlined by other optimization passes.
///
pub struct MergeSameExpressions;

impl OptimizerPass for MergeSameExpressions {
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
    ops: &RefCell<StableVec<Instruction>>,
    indices: &[usize],
    changes: &AtomicCounter,
) -> Option<()> {
    let mut op = ops.borrow()[idx].clone();
    let out = op.out?;
    let mut is_mut = false;
    opt.visit_operation(&mut op.operation, &mut Some(out), |_, var| {
        if !var.is_immutable() {
            is_mut = true;
        }
    });
    opt.visit_out(&mut op.out, |_, var| {
        if !var.is_immutable() {
            is_mut = true;
        }
    });
    if is_mut {
        return None;
    }
    for rhs_idx in indices.iter().skip(i + 1) {
        if op.operation == ops.borrow()[*rhs_idx].operation {
            ops.borrow_mut()[*rhs_idx].operation = Operation::Copy(out);
            changes.inc();
        }
    }
    Some(())
}
