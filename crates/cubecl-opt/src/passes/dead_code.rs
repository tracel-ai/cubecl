use std::{
    mem::transmute,
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

use cubecl_core::ir::{ConstantScalarValue, Variable};
use petgraph::visit::EdgeRef;

use crate::{visit_noop, AtomicCounter, ControlFlow, Optimizer};

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

pub struct EliminateConstBranches;

impl OptimizationPass for EliminateConstBranches {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        for block in opt.node_ids() {
            let control_flow = opt.program[block].control_flow.clone();
            let current = control_flow.borrow().clone();
            match current {
                ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    ..
                } if cond.as_const().is_some() => {
                    let cond = cond.as_const().unwrap().as_bool();
                    let mut edges = opt.program.edges(block);
                    if cond {
                        let edge = edges.find(|it| it.target() == or_else).unwrap().id();
                        opt.program.remove_edge(edge);
                    } else {
                        let edge = edges.find(|it| it.target() == then).unwrap().id();
                        opt.program.remove_edge(edge);
                    }
                    *control_flow.borrow_mut() = ControlFlow::None;
                    changes.inc();
                }
                ControlFlow::Switch {
                    value,
                    default,
                    branches,
                    ..
                } if value.as_const().is_some() => {
                    let value = match value.as_const().unwrap() {
                        ConstantScalarValue::Int(val, _) => unsafe {
                            transmute::<i32, u32>(val as i32)
                        },
                        ConstantScalarValue::UInt(val) => val as u32,
                        _ => unreachable!("Switch cases must be integer"),
                    };
                    let branch = branches.into_iter().find(|(val, _)| *val == value);
                    let branch = branch.map(|it| it.1).unwrap_or(default);
                    let edges = opt.program.edges(block).filter(|it| it.target() != branch);
                    let edges: Vec<_> = edges.map(|it| it.id()).collect();
                    for edge in edges {
                        opt.program.remove_edge(edge);
                    }
                    *control_flow.borrow_mut() = ControlFlow::None;
                    changes.inc();
                }
                _ => {}
            }
        }
    }
}

pub struct EliminateDeadBlocks;

impl OptimizationPass for EliminateDeadBlocks {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        while search_dead_blocks(opt) {
            changes.inc();
        }
    }
}

fn search_dead_blocks(opt: &mut Optimizer) -> bool {
    for block in opt.node_ids() {
        if block != opt.entry() && opt.predecessors(block).is_empty() {
            let edges: Vec<_> = opt.program.edges(block).map(|it| it.id()).collect();
            for edge in edges {
                opt.program.remove_edge(edge);
            }
            opt.program.remove_node(block);
            return true;
        }
    }

    false
}
