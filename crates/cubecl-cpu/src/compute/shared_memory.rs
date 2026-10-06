use cubecl_llvm::SharedMemories;
use cubecl_server::{
    memory_management::{ErrorGraph, MemoryManagement, PageUpdate},
    storage::BytesStorage,
};

/// Reserves the shared memory of a launch out of the stream's dedicated pool, and writes each
/// block into the slot of `table` the kernel reads it from. Those slots follow the buffers, so
/// the table is padded first when the kernel takes fewer buffers than the launch provides.
///
/// Blocks requiring more than the pool's SIMD alignment are over-reserved and their bases
/// rounded up.
///
/// The reservations are released right away: shared-memory launches never overlap — the stream
/// drains before enqueuing one (see `CpuStream::enqueue_task`).
pub fn reserve_shared_memories(
    memory: &mut MemoryManagement<BytesStorage>,
    failures: &mut ErrorGraph,
    shared_memories: &SharedMemories,
    table: &mut Vec<*mut std::ffi::c_void>,
) {
    let end = shared_memories.base + shared_memories.blocks.len();
    if table.len() < end {
        table.resize(end, core::ptr::null_mut());
    }

    // The handles are held until every block is reserved: releasing one right away would let
    // the pool hand the same memory out to the next shared memory of the same launch.
    let mut handles = Vec::with_capacity(shared_memories.blocks.len());

    for (slot, block) in shared_memories.blocks.iter().enumerate() {
        let padding = if block.align > BytesStorage::ALIGNMENT {
            block.align - 1
        } else {
            0
        };
        let handle = memory
            .reserve((block.size + padding) as u64, PageUpdate::Allow, failures)
            .expect("Failed to reserve the shared memory of the launch");
        let reserved = memory
            .get_resource(handle.clone().binding(), None, None)
            .expect("Failed to resolve the shared memory of the launch");
        handles.push(handle);

        let (ptr, _) = reserved.get_write_ptr_and_length();
        let ptr = ptr.wrapping_add(ptr.align_offset(block.align));
        table[shared_memories.base + slot] = ptr as *mut std::ffi::c_void;
    }
}
