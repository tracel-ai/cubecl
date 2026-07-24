use cubecl_core::ir::{AddressSpace, Type};
use cubecl_opt::Optimizer;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct SharedMemory {
    pub id: u32,
    pub ty: Type,
    // Length include the vectorization factor
    pub length: usize,
}

#[derive(Default, Debug)]
pub struct SharedMemories(pub Vec<SharedMemory>);

impl SharedMemories {
    pub fn visit(&mut self, opt: &Optimizer) {
        for (id, memory) in &opt.main.memories {
            if matches!(memory.address_space, AddressSpace::Shared) {
                let elem = memory.value_ty.scalar_value_type();
                let length = memory.value_ty.size() / elem.size();

                self.0.push(SharedMemory {
                    id: *id,
                    ty: elem,
                    length,
                })
            }
        }
        self.0.sort_by_key(|a| a.id);
    }
}
