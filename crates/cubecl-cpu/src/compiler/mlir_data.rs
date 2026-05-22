use super::passes::shared_memories::SharedMemories;
use crate::{
    compiler::{builtin::BuiltinArray, memref::LineMemRef},
    compute::schedule::BindingsResource,
};
use cubecl_core::CubeDim;
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::sync::{
    Arc,
    atomic::{AtomicI32, Ordering},
};

pub const SYNC_BARRIER_COUNTER_INDEX: usize = 0;
pub const SYNC_STOPPED_COUNTER_INDEX: usize = 1;
pub const SYNC_BARRIER_TARGET_INDEX: usize = 2;
pub const SYNC_CURRENT_CUBE_DIM_INDEX: usize = 3;

pub struct SyncCubeState {
    pub atomics: [AtomicI32; Self::len()],
}

impl SyncCubeState {
    pub const fn len() -> usize {
        4
    }

    fn new(cube_dim_size: i32) -> Self {
        Self {
            atomics: [
                AtomicI32::new(0),
                AtomicI32::new(0),
                AtomicI32::new(cube_dim_size),
                AtomicI32::new(cube_dim_size),
            ],
        }
    }
}

pub struct SharedMlirData {
    pub args_zero_indirection: Vec<LineMemRef>,
    pub info: Vec<u64>,
    pub args_first_indirection: Vec<*mut ()>,
    pub sync_cube_state: Box<SyncCubeState>,
}

unsafe impl Send for SharedMlirData {}
unsafe impl Sync for SharedMlirData {}

pub struct MlirData {
    pub shared_mlir_data: Arc<SharedMlirData>,
    pub args_second_indirection: Vec<*mut ()>,
    pub builtin: BuiltinArray,
}

unsafe impl Send for MlirData {}

impl Clone for MlirData {
    fn clone(&self) -> Self {
        Self {
            shared_mlir_data: Arc::clone(&self.shared_mlir_data),
            args_second_indirection: self.args_second_indirection.clone(),
            builtin: self.builtin.clone(),
        }
    }
}

impl MlirData {
    pub fn new(
        bindings: BindingsResource,
        shared_memories: &SharedMemories,
        memory_management_shared_memory: &mut MemoryManagement<BytesStorage>,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
    ) -> Self {
        let BindingsResource { resources, info } = bindings;
        let cube_dim_size = cube_dim.num_elems() as i32;

        let builtin = BuiltinArray::new(cube_dim, cube_count);
        let indirect_args_len = resources.len() + shared_memories.0.len() + 2;
        let total_args_len = indirect_args_len + BuiltinArray::len();

        let args_zero_indirection = Vec::with_capacity(indirect_args_len);
        let args_first_indirection = Vec::with_capacity(indirect_args_len);
        let mut args_second_indirection = Vec::with_capacity(total_args_len);
        let info = info.data;

        let mut shared_mlir_data = SharedMlirData {
            args_zero_indirection,
            args_first_indirection,
            info,
            sync_cube_state: Box::new(SyncCubeState::new(cube_dim_size)),
        };

        let mut push_undirected = |line_memref: LineMemRef| {
            shared_mlir_data.args_zero_indirection.push(line_memref);
            let undirected =
                shared_mlir_data.args_zero_indirection.last_mut().unwrap() as *mut LineMemRef;
            shared_mlir_data
                .args_first_indirection
                .push(undirected as *mut ());
            let undirected =
                shared_mlir_data.args_first_indirection.last_mut().unwrap() as *mut *mut ();

            args_second_indirection.push(undirected as *mut ());
        };

        for resource in resources {
            let (ptr, len) = resource.get_write_ptr_and_length();
            let line_memref = LineMemRef::new(ptr, len);
            push_undirected(line_memref);
        }

        let mut smem_handles = Vec::with_capacity(shared_memories.0.len());
        for shared_memory in shared_memories.0.iter() {
            let length_bytes = shared_memory.ty.size() as u64 * shared_memory.length as u64;
            let handle = memory_management_shared_memory
                .reserve(length_bytes)
                .unwrap();

            smem_handles.push(handle.clone());

            let handle = memory_management_shared_memory
                .get_resource(handle.binding(), None, None)
                .expect("Failed to find resource");
            let (ptr, len) = handle.get_write_ptr_and_length();
            let line_memref = LineMemRef::new(ptr, len);
            push_undirected(line_memref);
        }
        // It is important to make sure multiple shared memories don't shared the same handle.
        core::mem::drop(smem_handles);

        let ptr = shared_mlir_data.info.as_mut_ptr() as *mut u8;
        let info_len_bytes = shared_mlir_data.info.len() * core::mem::size_of::<u64>();
        let line_memref = LineMemRef::new(ptr, info_len_bytes);
        push_undirected(line_memref);

        let sync_cube_state = shared_mlir_data.sync_cube_state.atomics.as_mut_ptr() as *mut u8;
        push_undirected(LineMemRef::new(sync_cube_state, SyncCubeState::len()));

        let shared_mlir_data = Arc::new(shared_mlir_data);

        Self {
            shared_mlir_data,
            args_second_indirection,
            builtin,
        }
    }

    pub fn push_builtin(&mut self) {
        for arg in self.builtin.0.iter_mut() {
            self.args_second_indirection
                .push(arg as *mut u32 as *mut ());
        }
    }

    pub fn complete_unit(&self) {
        self.shared_mlir_data.sync_cube_state.atomics[SYNC_CURRENT_CUBE_DIM_INDEX]
            .fetch_sub(1, Ordering::AcqRel);
    }
}
