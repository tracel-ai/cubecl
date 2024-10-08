use std::{collections::HashMap, mem::take};

use cubecl_core::ir::{
    BinaryOperator, Item, LineInitOperator, Operation, Operator, UnaryOperator, Variable,
};
use stable_vec::StableVec;

use crate::{AtomicCounter, Optimizer};

use super::OptimizerPass;

/// Merge consecutive index assigns into a vectorized value into a single constructor call.
/// For example, in `wgsl`:
/// ```ignore
/// a[0] = 1;
/// a[1] = 2;
/// a[2] = 3;
/// a[4] = 4;
/// ```
/// would become
/// ```ignore
/// a = vec4(1, 2, 3, 4);
/// ```
/// This is more efficient particularly in SSA form.
///
pub struct CompositeMerge;

impl OptimizerPass for CompositeMerge {
    fn apply_pre_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        let blocks = opt.node_ids();

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

/// Remove indices into scalar values that are left over from iterating over a dynamically vectorized
/// value. This simplifies the backend by removing the need to specially handle this case.
pub struct RemoveIndexScalar;

impl OptimizerPass for RemoveIndexScalar {
    fn apply_post_ssa(&mut self, opt: &mut Optimizer, changes: AtomicCounter) {
        let blocks = opt.node_ids();

        for block in blocks {
            let ops = opt.program[block].ops.clone();
            for op in ops.borrow_mut().values_mut() {
                if let Operation::Operator(Operator::Index(BinaryOperator { lhs, rhs, out })) = op {
                    if !lhs.is_array() {
                        if let Some(index) = rhs.as_const() {
                            let index = index.as_u32();
                            let vectorization =
                                lhs.item().vectorization.map(|it| it.get()).unwrap_or(1);
                            if vectorization == 1 {
                                assert_eq!(index, 0, "Can't index into scalar");
                                *op = Operator::Assign(UnaryOperator {
                                    input: *lhs,
                                    out: *out,
                                })
                                .into();
                                changes.inc();
                            }
                        }
                    }
                }
            }
        }
    }
}
