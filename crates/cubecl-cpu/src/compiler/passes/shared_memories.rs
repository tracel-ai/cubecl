use cubecl_core::ir::{Elem, Variable, VariableKind};
use cubecl_opt::Optimizer;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory {
    pub id: u32,
    pub elem: Elem,
    // Length include the vectorization factor
    pub length: u32,
}

#[derive(Default)]
pub struct SharedMemories(pub Vec<SharedMemory>);

impl SharedMemories {
    pub fn visit_variable(&mut self, variable: Variable) {
        // Alignment is ignored for the moment it is taken from the type
        let VariableKind::SharedMemory { id, length, .. } = variable.kind else {
            return;
        };
        if self.0.iter().all(|shared_memory| shared_memory.id != id) {
            let elem = variable.elem();
            let vectorization = variable.vectorization_factor();
            let length = length * vectorization as u32;
            self.0.push(SharedMemory { id, elem, length });
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
            }
        }
        self.0.sort_by(|a, b| a.id.cmp(&b.id));
    }
}
