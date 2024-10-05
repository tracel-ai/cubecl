use std::{
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

use cubecl_core::ir::Variable;

use crate::{visit_noop, AtomicCounter, Optimizer};

use super::OptimizationPass;

pub struct EliminateUnusedVariables;

impl OptimizationPass for EliminateUnusedVariables {
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
            let mut op = opt.program[node].ops.borrow()[idx].clone();
            let mut out = None;
            let used = Rc::new(AtomicBool::new(false));
            opt.visit_operation(&mut op, visit_noop, |_, var| {
                // Exclude outputs
                if !matches!(
                    var,
                    Variable::GlobalOutputArray { .. } | Variable::Slice { .. }
                ) {
                    out = Some(*var);
                }
            });
            if let Some(out) = out {
                let used = used.clone();
                opt.visit_all(
                    |_, var| {
                        if *var == out {
                            used.store(true, Ordering::Release);
                        }
                    },
                    visit_noop,
                );
                if !used.load(Ordering::Acquire) {
                    opt.program[node].ops.borrow_mut().remove(idx);
                    return true;
                }
            }
        }
    }

    false
}
