use alloc::vec::Vec;
use cubecl_ir::{AddressSpace, CopyMemoryOperands, Instruction, Memory, Operation, Type, Value};
use hashbrown::{HashMap, HashSet};

use crate::{
    AtomicCounter, Function, GlobalState,
    analyses::{integer_range::var_id, pointer_source::PointerSource},
    visit_noop,
};

use super::OptimizerPass;

/// Transform consecutive `Index` and `IndexAssign` operations that don't modify the value into a
/// single `Copy` operation.
pub struct CopyTransform;

impl OptimizerPass for CopyTransform {
    fn apply_post_ssa(&mut self, func: &mut Function, state: &GlobalState, changes: AtomicCounter) {
        let ptr_source = func.analysis::<PointerSource>(state);
        for block in func.node_ids() {
            let mut reads = HashMap::new();
            let mut writes = HashMap::new();
            let ops = func[block].ops.clone();
            let indices = ops.borrow().indices().collect::<Vec<_>>();
            for idx in indices {
                let inst = ops.borrow()[idx].clone();
                match &inst.operation {
                    Operation::Memory(Memory::Load(ptr))
                        if is_copyable(&ptr.ty)
                            && let Some(source) = ptr_source.get(ptr)
                            && source.value_ty == inst.ty()
                            && !is_reused(func, state, &inst.out) =>
                    {
                        if let Some(id) = var_id(&inst.out()) {
                            reads.insert(id, (idx, *ptr, source.root_ptr));
                        }
                    }
                    Operation::Memory(Memory::Store(op))
                        if is_copyable(&op.ptr.ty)
                            && let Some(source) = ptr_source.get(&op.ptr) =>
                    {
                        if let Some(id) = var_id(&op.value) {
                            writes.insert(id, (idx, op.ptr, source.root_ptr));
                        }
                    }
                    _ => {}
                }
            }
            let read_ids: HashSet<_> = reads.keys().collect();
            let write_ids: HashSet<_> = writes.keys().collect();
            let copy_ids = read_ids.intersection(&write_ids);
            for id in copy_ids {
                let (read_idx, in_ptr, in_source) = reads[*id];
                let (write_idx, out_ptr, out_source) = writes[*id];
                let mut is_overwritten = false;
                for mut inst in
                    (read_idx..write_idx).filter_map(|idx| ops.borrow().get(idx).cloned())
                {
                    func.visit_instruction(state, &mut inst, visit_noop, |_, var| {
                        if *var == in_source || *var == out_source {
                            is_overwritten = true;
                        }
                    });
                }
                if is_overwritten {
                    continue;
                }

                ops.borrow_mut().remove(read_idx);
                let copy = Memory::CopyMemory(CopyMemoryOperands {
                    source: in_ptr,
                    target: out_ptr,
                    len: 1,
                });
                ops.borrow_mut()[write_idx] = Instruction::no_out(copy);
                changes.inc();
            }
        }
    }
}

/// Copy is only implemented in SPIR-V for global -> shared or shared -> global. So no point merging
/// locals.
fn is_copyable(ty: &Type) -> bool {
    matches!(
        ty.address_space(),
        Some(AddressSpace::Global(..)) | Some(AddressSpace::Shared)
    )
}

fn is_reused(func: &mut Function, state: &GlobalState, var: &Option<Value>) -> bool {
    if let Some(var) = var.as_ref() {
        let count = AtomicCounter::new(0);
        func.visit_all(
            state,
            |_, other| {
                if other == var {
                    count.inc();
                }
            },
            |_, _| {},
        );
        count.get() > 1
    } else {
        false
    }
}
