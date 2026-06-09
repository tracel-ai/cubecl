use alloc::vec::Vec;
use cubecl_ir::{
    AddressSpace, Id, Instruction, Memory, Operation, Operator, Type, UnaryOperands, Variable,
    VariableKind,
};
use hashbrown::HashMap;

use crate::{
    AtomicCounter, Function, GlobalState,
    analyses::{pointer_source::PointerSource, writes::Writes},
    local_variable_id,
};

use super::OptimizerPass;

/// Split arrays with only constant indices into a set of local intermediates. This allows the
/// compiler to reorder them and optimize memory layout, along with enabling more inlining and
/// expression merging.
///
/// # Example
///
/// ```ignore
/// arr[0] = a;
/// arr[1] = b;
/// arr[2] = c;
/// ```
/// transforms to
/// ```ignore
/// let a1 = a;
/// let b1 = b;
/// let c1 = c;
/// ```
/// which can usually be completely merged out and inlined.
///
pub struct DisaggregateArray;

impl OptimizerPass for DisaggregateArray {
    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        let arrays = find_const_arrays(func);

        for Array { id, length, ty } in arrays {
            // Initialize in entry because we don't know where the array is actually declared.
            // The constant value will be inlined later so it doesn't matter as long as the
            // value is visible everywhere.
            let block = func.root;
            let old_insts = func[block].ops.take();
            let arr_id = id;
            let vars = (0..length)
                .map(|_| state.allocator.create_local_mut(ty.value_type()))
                .collect::<Vec<_>>();
            for var in &vars {
                let local_id = local_variable_id(var).unwrap();
                func.variables.insert(local_id, var.ty);
                let init =
                    Instruction::new(Operator::Cast(UnaryOperands { input: 0u32.into() }), *var);
                func[block].ops.borrow_mut().push(init);
            }
            func[block]
                .ops
                .borrow_mut()
                .extend(old_insts.into_iter().map(|it| it.1));
            replace_const_arrays(func, arr_id, &vars);
            func.invalidate_analysis::<PointerSource>();
            changes.inc();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Array {
    id: Id,
    length: usize,
    ty: Type,
}

fn find_const_arrays(func: &mut Function) -> Vec<Array> {
    let mut track_consts = HashMap::new();
    let mut arrays = HashMap::new();

    for block in func.node_ids() {
        let ops = func[block].ops.clone();
        for op in ops.borrow().values() {
            if let Operation::Memory(Memory::Index(index)) = &op.operation
                && let VariableKind::LocalMut { id } = index.list.kind
                && let Type::Array(ty, length, AddressSpace::Local) = index.list.ty
            {
                arrays.insert(
                    id,
                    Array {
                        id,
                        length,
                        ty: *ty,
                    },
                );
                let is_const = index.index.as_const().is_some();
                *track_consts.entry(id).or_insert(is_const) &= is_const;
            }
        }
    }

    track_consts
        .iter()
        .filter(|(_, is_const)| **is_const)
        .map(|(id, _)| arrays[id])
        .collect()
}

fn replace_const_arrays(func: &mut Function, arr_id: Id, vars: &[Variable]) {
    for block in func.node_ids() {
        let ops = func[block].ops.clone();
        for op in ops.borrow_mut().values_mut() {
            if let Operation::Memory(Memory::Index(index)) = &mut op.operation.clone()
                && let VariableKind::LocalMut { id, .. } = index.list.kind
                && id == arr_id
            {
                let const_index = index.index.as_const().unwrap().as_i64() as usize;
                op.operation = Operation::Memory(Memory::Reference(vars[const_index]));
                func.invalidate_analysis::<Writes>();
            }
        }
    }
}
