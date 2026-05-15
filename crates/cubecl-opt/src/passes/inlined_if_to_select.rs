use core::mem::take;

use alloc::vec::Vec;
use cubecl_ir::{Instruction, Operator, SelectOperands};
use petgraph::{graph::NodeIndex, visit::EdgeRef};

use crate::{AtomicCounter, ControlFlow, Function, GlobalState, passes::update_references};

use super::OptimizerPass;

/// Copy propagation sometimes leaves empty if-else branches because the assignments they contained
/// have been inlined into the following `phi` node, so the branches only serve to select a value
/// for the phi. This detects such cases and replaces the phi with select, merging the control flow
/// blocks into a single one.
pub struct EmptyBranchToSelect;

impl OptimizerPass for EmptyBranchToSelect {
    fn apply_post_ssa(&mut self, opt: &mut Function, _: &GlobalState, changes: AtomicCounter) {
        while run(opt) {
            changes.inc();
        }

        fn run(func: &mut Function) -> bool {
            for block in func.node_ids() {
                let control = { func[block].control_flow.borrow().clone() };
                if let ControlFlow::IfElse {
                    cond,
                    then,
                    or_else,
                    merge: Some(merge),
                } = control
                {
                    let then_empty = func[then].ops.borrow().is_empty()
                        && func[then].phi_nodes.borrow().is_empty();
                    let else_empty = func[or_else].ops.borrow().is_empty()
                        && func[or_else].phi_nodes.borrow().is_empty();

                    if then_empty
                        && else_empty
                        && is_simple(func, then, or_else, merge)
                        && func.predecessors(merge).len() == 2
                    {
                        let removed_phi = take(&mut *func[merge].phi_nodes.borrow_mut());
                        let mut selects = removed_phi
                            .into_iter()
                            .map(|phi| {
                                let then = phi.entries.iter().find(|it| it.block == then).unwrap();
                                let or_else =
                                    phi.entries.iter().find(|it| it.block == or_else).unwrap();
                                Instruction::new(
                                    Operator::Select(SelectOperands {
                                        cond,
                                        then: then.value,
                                        or_else: or_else.value,
                                    }),
                                    phi.out,
                                )
                            })
                            .collect::<Vec<Instruction>>();
                        selects.extend(
                            take(&mut *func[merge].ops.borrow_mut())
                                .into_iter()
                                .map(|(_, v)| v),
                        );
                        func[block].ops.borrow_mut().extend(selects);
                        let merge_successors = func.successors(merge);
                        let merge_control = func[merge].control_flow.borrow().clone();

                        let edges_to_remove = func
                            .edges(block)
                            .chain(func.edges(then))
                            .chain(func.edges(or_else))
                            .chain(func.edges(merge))
                            .map(|it| it.id())
                            .collect::<Vec<_>>();
                        for edge in edges_to_remove {
                            func.remove_edge(edge);
                        }
                        func.remove_node(then);
                        func.remove_node(or_else);
                        func.remove_node(merge);
                        for merge_successor in merge_successors {
                            func.add_edge(block, merge_successor, 0);
                        }
                        *func[block].control_flow.borrow_mut() = merge_control;
                        func.invalidate_structure();
                        update_references(func, merge, block);
                        return true;
                    }
                }
            }
            false
        }
    }
}

fn is_simple(func: &Function, then: NodeIndex, or_else: NodeIndex, merge: NodeIndex) -> bool {
    let no_control = matches!(*func[then].control_flow.borrow(), ControlFlow::None)
        && matches!(*func[or_else].control_flow.borrow(), ControlFlow::None);
    no_control && func.successors(then)[0] == merge && func.successors(or_else)[0] == merge
}
