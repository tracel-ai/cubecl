use core::{cell::RefCell, mem::take};

use alloc::vec::Vec;
use cubecl_ir::{CoopMma, Instruction, Operation, Operator, UnaryOperands};
use stable_vec::StableVec;

use crate::{AtomicCounter, Function, GlobalState, visit_noop};

use super::OptimizerPass;

/// Inline constants or simple reassignments that don't change the type. This simplifies the code
/// and makes it easier to find optimizable expressions.
pub struct InlineAssignments;

impl OptimizerPass for InlineAssignments {
    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        while search_loop(func, state) {
            changes.inc();
        }
    }
}

fn search_loop(func: &mut Function, state: &GlobalState) -> bool {
    for node in func.node_indices().collect::<Vec<_>>() {
        let mut removed_phi = Vec::new();
        // Remove trivial phi nodes left from PRE
        func[node].phi_nodes.borrow_mut().retain(|it| {
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

            let ops = take(&mut *func[node].ops.borrow_mut());
            phi_assigns.extend(ops.into_iter().map(|(_, v)| v));
            *func[node].ops.borrow_mut() = phi_assigns;
            return true;
        }

        let ops = func[node].ops.borrow().indices().collect::<Vec<_>>();

        for idx in ops {
            let op = func[node].ops.borrow()[idx].clone();
            match op.operation {
                Operation::Copy(input)
                | Operation::Operator(Operator::Cast(UnaryOperands { input }))
                | Operation::Operator(Operator::Reinterpret(UnaryOperands { input }))
                | Operation::CoopMma(CoopMma::Cast { input })
                    if (input.is_immutable() || input.is_array() || input.ty.is_ptr())
                        && (op.out().is_immutable()
                            || op.out().is_array()
                            || op.out().ty.is_ptr())
                        && input.ty == op.ty() =>
                {
                    func.visit_all(
                        state,
                        |_, var| {
                            if *var == op.out() {
                                *var = input
                            }
                        },
                        visit_noop,
                    );
                    func[node].ops.borrow_mut().remove(idx);
                    return true;
                }
                _ => {}
            }
        }
    }

    false
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
    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        for node in func.node_ids() {
            let ops = func[node].ops.clone();
            let indices = ops.borrow().indices().collect::<Vec<_>>();
            for (i, idx) in indices.iter().enumerate() {
                check_op(func, state, i, *idx, &ops, &indices, &changes);
            }
        }
    }
}

fn check_op(
    func: &mut Function,
    state: &GlobalState,
    i: usize,
    idx: usize,
    ops: &RefCell<StableVec<Instruction>>,
    indices: &[usize],
    changes: &AtomicCounter,
) -> Option<()> {
    let mut op = ops.borrow()[idx].clone();
    let out = op.out?;
    let mut is_mut = false;
    func.visit_operation(state, &mut op.operation, |_, var| {
        if !var.is_immutable() {
            is_mut = true;
        }
    });
    func.visit_out(&mut op.out, |_, var| {
        if !var.is_immutable() {
            is_mut = true;
        }
    });
    if is_mut {
        return None;
    }
    for rhs_idx in indices.iter().skip(i + 1) {
        // Type needs to be checked because versioned variable can have the same expression, but different output
        if op.operation == ops.borrow()[*rhs_idx].operation
            && out.ty == ops.borrow()[*rhs_idx].out().ty
        {
            ops.borrow_mut()[*rhs_idx].operation = Operation::Copy(out);
            changes.inc();
        }
    }
    Some(())
}
