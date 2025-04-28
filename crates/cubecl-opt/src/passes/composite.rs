use std::{collections::HashMap, mem::take};

use cubecl_ir::{
    Id, IndexAssignOperator, IndexOperator, Instruction, Item, LineInitOperator, Operation,
    Operator, Variable, VariableKind,
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
            let mut assigns = HashMap::<Id, Vec<(usize, u32, Variable)>>::new();

            let ops = opt.program[block].ops.clone();
            let indices = { ops.borrow().indices().collect::<Vec<_>>() };
            for idx in indices {
                // Reset writes when read
                {
                    let op = &mut ops.borrow_mut()[idx];
                    opt.visit_operation(&mut op.operation, &mut op.out, |opt, var| {
                        if let Some(id) = opt.local_variable_id(var) {
                            assigns.remove(&id);
                        }
                    });
                }

                let op = { ops.borrow()[idx].clone() };
                if let (
                    Operation::Operator(Operator::IndexAssign(IndexAssignOperator {
                        index,
                        value,
                        ..
                    })),
                    Some(VariableKind::LocalMut { id }),
                ) = (op.operation, op.out.map(|it| it.kind))
                {
                    let item = op.out.unwrap().item;
                    if let Some(index) = index.as_const() {
                        let index = index.as_u32();
                        let vectorization = item.vectorization.map(|it| it.get()).unwrap_or(1);
                        if vectorization > 1 {
                            let assigns = assigns.entry(id).or_default();
                            assigns.push((idx, index, value));
                            if assigns.len() as u8 == vectorization {
                                merge_assigns(
                                    &mut opt.program[block].ops.borrow_mut(),
                                    take(assigns),
                                    id,
                                    item,
                                );
                                changes.inc();
                            }
                        } else {
                            assert_eq!(index, 0, "Can't index into scalar");
                            opt.program[block].ops.borrow_mut()[idx] = Instruction::new(
                                Operation::Copy(value),
                                Variable::new(VariableKind::LocalMut { id }, item),
                            )
                        }
                    }
                }
            }
        }
    }
}

fn merge_assigns(
    ops: &mut StableVec<Instruction>,
    mut assigns: Vec<(usize, u32, Variable)>,
    id: Id,
    item: Item,
) {
    for assignment in assigns.iter() {
        ops.remove(assignment.0);
    }
    let last = assigns.iter().map(|it| it.0).max().unwrap();
    assigns.sort_by_key(|it| it.1);
    let inputs = assigns.iter().map(|it| it.2).collect::<Vec<_>>();
    let out = Variable::new(VariableKind::LocalMut { id }, item);
    ops.insert(
        last,
        Instruction::new(Operator::InitLine(LineInitOperator { inputs }), out),
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
                if let Operation::Operator(Operator::Index(IndexOperator { list, index, .. })) =
                    &mut op.operation
                {
                    if !list.is_array() {
                        if let Some(index) = index.as_const() {
                            let index = index.as_u32();
                            let vectorization =
                                list.item.vectorization.map(|it| it.get()).unwrap_or(1);
                            if vectorization == 1 {
                                assert_eq!(index, 0, "Can't index into scalar");
                                op.operation = Operation::Copy(*list);
                                changes.inc();
                            }
                        }
                    }
                }
            }
        }
    }
}
