use std::mem::take;

use cubecl_ir::{Instruction, Operator, Select};
use petgraph::{graph::NodeIndex, visit::EdgeRef};

use crate::{AtomicCounter, ControlFlow, Optimizer, passes::update_references};

use super::OptimizerPass;

/// Copy propagation sometimes leaves empty if-else branches because the assignments they contained
/// have been inlined into the following `phi` node, so the branches only serve to select a value
/// for the phi. This detects such cases and replaces the phi with select, merging the control flow
/// blocks into a single one.
pub struct EmptyBranchToSelect;

impl OptimizerPass for EmptyBranchToSelect {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        while run(opt) {
            changes.inc();
        }

        fn run(opt: &mut Optimizer) -> bool {
            for block in opt.node_ids() {
                let control = { opt.program[block].control_flow.borrow().clone() };
                if let ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    merge: Some(merge),
                } = control
                {
                    let then_empty = opt.program[then].ops.borrow().is_empty()
                        && opt.program[then].phi_nodes.borrow().is_empty();
                    let else_empty = opt.program[or_else].ops.borrow().is_empty()
                        && opt.program[or_else].phi_nodes.borrow().is_empty();

                    if then_empty
                        && else_empty
                        && is_simple(opt, then, or_else, merge)
                        && opt.predecessors(merge).len() == 2
                    {
                        let removed_phi = take(&mut *opt.program[merge].phi_nodes.borrow_mut());
                        let mut selects = removed_phi
                            .into_iter()
                            .map(|phi| {
                                let then = phi.entries.iter().find(|it| it.block == then).unwrap();
                                let or_else =
                                    phi.entries.iter().find(|it| it.block == or_else).unwrap();
                                Instruction::new(
                                    Operator::Select(Select {
                                        cond,
                                        then: then.value,
                                        or_else: or_else.value,
                                    }),
                                    phi.out,
                                )
                            })
                            .collect::<Vec<Instruction>>();
                        selects.extend(
                            take(&mut *opt.program[merge].ops.borrow_mut())
                                .into_iter()
                                .map(|(_, v)| v),
                        );
                        opt.program[block].ops.borrow_mut().extend(selects);
                        let merge_successors = opt.successors(merge);
                        let merge_control = opt.program[merge].control_flow.borrow().clone();

                        let edges_to_remove = opt
                            .program
                            .edges(block)
                            .chain(opt.program.edges(then))
                            .chain(opt.program.edges(or_else))
                            .chain(opt.program.edges(merge))
                            .map(|it| it.id())
                            .collect::<Vec<_>>();
                        for edge in edges_to_remove {
                            opt.program.remove_edge(edge);
                        }
                        opt.program.remove_node(then);
                        opt.program.remove_node(or_else);
                        opt.program.remove_node(merge);
                        for merge_successor in merge_successors {
                            opt.program.add_edge(block, merge_successor, 0);
                        }
                        *opt.program[block].control_flow.borrow_mut() = merge_control;
                        opt.invalidate_structure();
                        update_references(opt, merge, block);
                        return true;
                    }
                }
            }
            false
        }
    }
}

fn is_simple(opt: &Optimizer, then: NodeIndex, or_else: NodeIndex, merge: NodeIndex) -> bool {
    let no_control = matches!(*opt.program[then].control_flow.borrow(), ControlFlow::None)
        && matches!(
            *opt.program[or_else].control_flow.borrow(),
            ControlFlow::None
        );
    no_control && opt.successors(then)[0] == merge && opt.successors(or_else)[0] == merge
}
