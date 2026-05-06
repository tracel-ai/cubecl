use cubecl_core::ir::{OperationReflect, StorageType, Variable, VariableKind};
use cubecl_opt::Optimizer;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SharedMemory {
    pub id: u32,
    pub ty: StorageType,
    // Length include the vectorization factor
    pub length: usize,
}

#[derive(Default)]
pub struct SharedMemories(pub Vec<SharedMemory>);

impl SharedMemories {
    pub fn visit_variable(&mut self, variable: Variable) {
        // Alignment is ignored for the moment it is taken from the type
        match variable.kind {
            VariableKind::Shared { id, .. }
                if self.0.iter().all(|shared_memory| shared_memory.id != id) =>
            {
                let elem = variable.storage_type();
                let length = variable.ty.size() / elem.size();

                self.0.push(SharedMemory {
                    id,
                    ty: elem,
                    length,
                });
            }
            _ => {}
        }
    }
    pub fn visit(&mut self, opt: &Optimizer) {
        for node in opt.main.node_indices().collect::<Vec<_>>() {
            let phi = opt.main[node].phi_nodes.clone();
            let ops = opt.main[node].ops.clone();

            for phi in phi.borrow_mut().iter_mut() {
                self.visit_variable(phi.out);
            }
            for op in ops.borrow_mut().values_mut() {
                if let Some(out) = op.out {
                    self.visit_variable(out);
                }
                if let Some(args) = op.operation.args() {
                    for arg in args {
                        self.visit_variable(arg);
                    }
                }
            }
        }
        self.0.sort_by_key(|a| a.id);
    }
}
