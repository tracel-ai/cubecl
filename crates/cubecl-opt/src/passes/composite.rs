use std::{collections::HashMap, mem::take};

use cubecl_core::ir::{
    BinaryOperator, Item, LineInitOperator, Operation, Operator, UnaryOperator, Variable,
};
use stable_vec::StableVec;

use crate::{AtomicCounter, Optimizer};

use super::OptimizationPass;

pub struct CompositeMerge;

impl OptimizationPass for CompositeMerge {
    fn apply_pre_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        let blocks = opt.program.node_indices().collect::<Vec<_>>();

        for block in blocks {
            let mut assigns = HashMap::<(u16, u8), Vec<(usize, u32, Variable)>>::new();

            let ops = opt.program[block].ops.clone();
            let indices = { ops.borrow().indices().collect::<Vec<_>>() };
            for idx in indices {
                // Reset writes when read
                opt.visit_operation(
                    &mut ops.borrow_mut()[idx],
                    |opt, var| {
                        if let Some(id) = opt.local_variable_id(var) {
                            assigns.remove(&id);
                        }
                    },
                    |_, _| {},
                );

                let op = { ops.borrow()[idx].clone() };
                if let Operation::Operator(Operator::IndexAssign(BinaryOperator {
                    lhs,
                    rhs,
                    out: Variable::Local { id, depth, item },
                })) = op
                {
                    if let Some(index) = lhs.as_const() {
                        let index = index.as_u32();
                        let vectorization = item.vectorization.map(|it| it.get()).unwrap_or(1);
                        if vectorization > 1 {
                            let assigns = assigns.entry((id, depth)).or_default();
                            assigns.push((idx, index, rhs));
                            if assigns.len() as u8 == vectorization {
                                merge_assigns(
                                    &mut opt.program[block].ops.borrow_mut(),
                                    take(assigns),
                                    id,
                                    depth,
                                    item,
                                );
                                changes.inc();
                            }
                        } else {
                            assert_eq!(index, 0, "Can't index into scalar");
                            opt.program[block].ops.borrow_mut()[idx] =
                                Operator::Assign(UnaryOperator {
                                    input: rhs,
                                    out: Variable::Local { id, item, depth },
                                })
                                .into()
                        }
                    }
                }
            }
        }
    }
}

fn merge_assigns(
    ops: &mut StableVec<Operation>,
    mut assigns: Vec<(usize, u32, Variable)>,
    id: u16,
    depth: u8,
    item: Item,
) {
    for assignment in assigns.iter() {
        ops.remove(assignment.0);
    }
    let last = assigns.iter().map(|it| it.0).max().unwrap();
    assigns.sort_by_key(|it| it.1);
    let inputs = assigns.iter().map(|it| it.2).collect::<Vec<_>>();
    let out = Variable::Local { id, item, depth };
    ops.insert(
        last,
        Operation::Operator(Operator::InitLine(LineInitOperator { out, inputs })),
    );
}
