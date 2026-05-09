use alloc::vec::Vec;
use cubecl_ir::{Memory, Operation, Operator, Variable, VariableKind};

use crate::{analyses::pointer_source::PointerSource, passes::OptimizerPass};

/// Inline local references that are only used in the local scope. Keeps them for function calls.
pub struct InlineRef;

impl OptimizerPass for InlineRef {
    fn apply_pre_ssa(
        &mut self,
        func: &mut crate::Function,
        state: &crate::GlobalState,
        changes: crate::AtomicCounter,
    ) {
        apply(func, state, changes);
    }

    fn apply_post_ssa(
        &mut self,
        func: &mut crate::Function,
        state: &crate::GlobalState,
        changes: crate::AtomicCounter,
    ) {
        apply(func, state, changes);
    }
}

fn apply(func: &mut crate::Function, state: &crate::GlobalState, changes: crate::AtomicCounter) {
    let ptr_source = func.analysis::<PointerSource>(state);
    let func_params = func.all_params().collect::<Vec<_>>();
    for block in func.node_ids() {
        for inst in func.block_mut(block).ops.borrow_mut().values_mut() {
            if let Operation::Memory(memory) = inst.operation.clone() {
                match memory {
                    Memory::Load(var) if is_local(ptr_source.get(&var), &func_params) => {
                        let local = ptr_source.get(&var).unwrap();
                        inst.operation = Operation::Copy(local);
                        changes.inc();
                    }
                    Memory::Store(op) if is_local(ptr_source.get(&op.ptr), &func_params) => {
                        let local = ptr_source.get(&op.ptr).unwrap();
                        inst.out = Some(local);
                        inst.operation = Operation::Copy(op.value);
                        changes.inc();
                    }
                    _ => {}
                }
            }
            if let Operation::Operator(Operator::InsertComponent(_)) = inst.operation.clone()
                && is_local(ptr_source.get(&inst.out()), &func_params)
            {
                let local = ptr_source.get(&inst.out()).unwrap();
                inst.out = Some(local);
                changes.inc();
            }
        }
    }
}

fn is_local(var: Option<Variable>, params: &[Variable]) -> bool {
    var.is_some_and(|var| {
        matches!(var.kind, VariableKind::LocalMut { .. })
            && !var.is_array()
            && !params.contains(&var)
    })
}
