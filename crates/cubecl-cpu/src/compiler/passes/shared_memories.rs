use cubecl_core::ir::{OperationReflect, StorageType, Variable, VariableKind};
use cubecl_opt::Optimizer;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SharedMemory {
    Array {
        id: u32,
        ty: StorageType,
        // Length include the vectorization factor
        length: usize,
    },
    Value {
        id: u32,
        ty: StorageType,
    },
}

impl SharedMemory {
    pub fn id(&self) -> u32 {
        match self {
            SharedMemory::Array { id, .. } => *id,
            SharedMemory::Value { id, .. } => *id,
        }
    }
}

#[derive(Default)]
pub struct SharedMemories(pub Vec<SharedMemory>);

impl SharedMemories {
    pub fn visit_variable(&mut self, variable: Variable) {
        // Alignment is ignored for the moment it is taken from the type
        match variable.kind {
            VariableKind::SharedArray { id, length, .. } => {
                if self.0.iter().all(|shared_memory| shared_memory.id() != id) {
                    let elem = variable.storage_type();
                    let vectorization = variable.line_size();
                    let length = length * vectorization;
                    self.0.push(SharedMemory::Array {
                        id,
                        ty: elem,
                        length,
                    });
                }
            }
            VariableKind::Shared { id } => {
                if self.0.iter().all(|shared_memory| shared_memory.id() != id) {
                    let elem = variable.storage_type();
                    self.0.push(SharedMemory::Value { id, ty: elem });
                }
            }
            _ => {}
        }
    }
    pub fn visit(&mut self, opt: &Optimizer) {
        for node in opt.program.node_indices().collect::<Vec<_>>() {
            let phi = opt.program[node].phi_nodes.clone();
            let ops = opt.program[node].ops.clone();

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
        self.0.sort_by_key(|a| a.id());
    }
}
