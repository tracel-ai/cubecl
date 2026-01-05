use cubecl_core::ir::Type;
use cubecl_opt::SharedLiveness;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SharedMemory {
    Array {
        id: u32,
        ty: Type,
        // Length includes unroll_factor; vectorization is in ty.size()
        length: u32,
        offset: u32,
    },
    Value {
        id: u32,
        ty: Type,
        offset: u32,
    },
}

impl SharedMemory {
    pub fn id(&self) -> u32 {
        match self {
            SharedMemory::Array { id, .. } => *id,
            SharedMemory::Value { id, .. } => *id,
        }
    }

    pub fn offset(&self) -> u32 {
        match self {
            SharedMemory::Array { offset, .. } => *offset,
            SharedMemory::Value { offset, .. } => *offset,
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            SharedMemory::Array { ty, length, .. } => *length * ty.size() as u32,
            SharedMemory::Value { ty, .. } => ty.size() as u32,
        }
    }
}

#[derive(Default)]
pub struct SharedMemories(pub Vec<SharedMemory>);

impl SharedMemories {
    /// Build from the [SharedLiveness] analysis so non-overlapping lifetimes can reuse memory.
    pub fn from_liveness(shared_liveness: &SharedLiveness) -> Self {
        let mut memories: Vec<SharedMemory> = shared_liveness
            .allocations
            .values()
            .map(|alloc| match alloc.smem {
                cubecl_opt::SharedMemory::Array { id, length, ty, .. } => SharedMemory::Array {
                    id,
                    ty,
                    length,
                    offset: alloc.offset,
                },
                cubecl_opt::SharedMemory::Value { id, ty, .. } => SharedMemory::Value {
                    id,
                    ty,
                    offset: alloc.offset,
                },
            })
            .collect();

        memories.sort_by_key(|m| m.id());
        Self(memories)
    }

    pub fn size(&self) -> Option<u64> {
        self.0.iter().map(|m| (m.offset() + m.size()) as u64).max()
    }
}
