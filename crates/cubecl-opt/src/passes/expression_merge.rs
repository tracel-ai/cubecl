use cubecl_core::ir::{Operation, Operator};

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
