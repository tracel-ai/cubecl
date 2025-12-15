use super::passes::shared_memories::SharedMemories;
use crate::{
    compiler::{builtin::BuiltinArray, memref::LineMemRef, passes::shared_memories::SharedMemory},
    compute::schedule::BindingsResource,
};
use cubecl_common::stream_id::StreamId;
use cubecl_core::server::{Handle, ScalarBinding};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};
use std::sync::Arc;

pub struct SharedMlirData {
    pub args_zero_indirection: Vec<LineMemRef>,
    pub metadata: Vec<u32>,
    pub args_first_indirection: Vec<*mut ()>,
    pub scalars: Vec<ScalarBinding>,
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
    ) -> Self {
        let BindingsResource {
            resources,
            scalars,
            metadata,
        } = bindings;

        let scalars_binding: Vec<_> = scalars.into_values().collect();

        let builtin = BuiltinArray::default();
        let max_buffer_size = resources.len() + scalars_binding.len() + BuiltinArray::len();

        let args_zero_indirection = Vec::with_capacity(max_buffer_size);
        let args_first_indirection = Vec::with_capacity(max_buffer_size);
        let mut args_second_indirection = Vec::with_capacity(max_buffer_size);
        let scalars: Vec<ScalarBinding> = Vec::with_capacity(max_buffer_size);
        let metadata = metadata.data;

        let mut shared_mlir_data = SharedMlirData {
            args_zero_indirection,
            args_first_indirection,
            scalars,
            metadata,
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

        for mut resource in resources {
            let ptr = resource.write();
            let line_memref = LineMemRef::new(ptr);
            push_undirected(line_memref);
        }

        let stream_id = StreamId::current();
        let mut smem_handles = Vec::with_capacity(shared_memories.0.len());
        for shared_memory in shared_memories.0.iter() {
            let (handle, length) = match shared_memory {
                SharedMemory::Array { ty, length, .. } => {
                    let length = (ty.size() * *length as usize) as u64;
                    let handle = memory_management_shared_memory.reserve(length).unwrap();
                    (handle, length)
                }
                SharedMemory::Value { ty, .. } => {
                    let length = ty.size() as u64;
                    let handle = memory_management_shared_memory.reserve(length).unwrap();
                    (handle, length)
                }
            };

            smem_handles.push(handle.clone());

            let b = Handle::new(handle, None, None, stream_id, 0, length).binding();
            let mut handle = memory_management_shared_memory
                .get_resource(b.memory, b.offset_start, b.offset_end)
                .expect("Failed to find resource");
            let ptr = handle.write();
            let line_memref = LineMemRef::new(ptr);
            push_undirected(line_memref);
        }
        // It is important to make sure multiple shared memories don't shared the same handle.
        core::mem::drop(smem_handles);

        let ptr = shared_mlir_data.metadata.as_mut();
        let line_memref = LineMemRef::new(ptr);
        push_undirected(line_memref);

        for scalar in scalars_binding {
            shared_mlir_data.scalars.push(scalar);
            let data = shared_mlir_data
                .scalars
                .last_mut()
                .unwrap()
                .data
                .as_mut_slice();
            let line_memref = LineMemRef::new(data);
            push_undirected(line_memref);
        }

        let shared_mlir_data = Arc::new(shared_mlir_data);

        Self {
            shared_mlir_data,
            args_second_indirection,
            builtin,
        }
    }

    pub fn push_builtin(&mut self) {
        for arg in self.builtin.dims.iter_mut() {
            self.args_second_indirection
                .push(arg as *mut u32 as *mut ());
        }
    }
}
