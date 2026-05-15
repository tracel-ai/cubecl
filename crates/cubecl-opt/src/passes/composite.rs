use core::mem::take;

use alloc::vec::Vec;
use cubecl_ir::{
    BinaryOperands, Id, InitVectorOperands, Instruction, Operation, Operator, Type, Variable,
    VariableKind, VectorInsertOperands,
};
use hashbrown::HashMap;
use stable_vec::StableVec;

use crate::{
    AtomicCounter, Function, GlobalState, analyses::pointer_source::PointerSource,
    local_variable_id,
};

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
    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        let blocks = func.node_ids();
        let ptr_source = func.analysis::<PointerSource>(state);

        for block in blocks {
            let mut assigns = HashMap::<Id, Vec<(usize, u32, Variable)>>::new();

            let ops = func[block].ops.clone();
            let indices = { ops.borrow().indices().collect::<Vec<_>>() };
            for idx in indices {
                // Reset writes when read
                {
                    let op = &mut ops.borrow_mut()[idx];
                    func.visit_operation(state, &mut op.operation, |_, var| {
                        if let Some(id) = local_variable_id(var) {
                            assigns.remove(&id);
                        }
                    });
                }

                let op = { ops.borrow()[idx].clone() };
                if let (
                    Operation::Operator(Operator::InsertComponent(VectorInsertOperands {
                        index,
                        value,
                        ..
                    })),
                    Some(VariableKind::LocalMut { id }),
                ) = (
                    op.operation,
                    op.out.map(|it| ptr_source.get(&it).unwrap_or(it).kind),
                ) && value.is_immutable()
                {
                    let item = op.out.unwrap().ty;
                    if let Some(index) = index.as_const() {
                        let index = index.as_u32();
                        let vector_size = item.vector_size();
                        if vector_size > 1 {
                            let assigns = assigns.entry(id).or_default();
                            assigns.push((idx, index, value));
                            if assigns.len() == vector_size {
                                merge_assigns(
                                    &mut func[block].ops.borrow_mut(),
                                    take(assigns),
                                    id,
                                    item,
                                );
                                func.variables.insert(id, item);
                                changes.inc();
                            }
                        } else {
                            assert_eq!(index, 0, "Can't index into scalar {}", op.out.unwrap());
                            func[block].ops.borrow_mut()[idx] = Instruction::new(
                                Operation::Copy(value),
                                Variable::new(VariableKind::LocalMut { id }, item),
                            );
                            func.variables.insert(id, item);
                            changes.inc();
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
    item: Type,
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
        Instruction::new(Operator::InitVector(InitVectorOperands { inputs }), out),
    );
}

/// Remove indices into scalar values that are left over from iterating over a dynamically vectorized
/// value. This simplifies the backend by removing the need to specially handle this case.
pub struct RemoveIndexScalar;

impl OptimizerPass for RemoveIndexScalar {
    fn apply_post_ssa(&mut self, func: &mut Function, _: &GlobalState, changes: AtomicCounter) {
        let blocks = func.node_ids();

        for block in blocks {
            let ops = func[block].ops.clone();
            for op in ops.borrow_mut().values_mut() {
                if let Operation::Operator(Operator::ExtractComponent(BinaryOperands {
                    lhs: vector,
                    rhs: index,
                    ..
                })) = &mut op.operation
                    && let Some(index) = index.as_const()
                {
                    let index = index.as_u32();
                    let vector_size = vector.ty.vector_size();
                    if vector_size == 1 {
                        assert_eq!(index, 0, "Can't index into scalar");
                        op.operation = Operation::Copy(*vector);
                        changes.inc();
                    }
                }
            }
        }
    }
}
