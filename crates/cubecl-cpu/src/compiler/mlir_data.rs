use std::sync::Arc;

use cubecl_core::server::{Bindings, Handle, ScalarBinding};
use cubecl_runtime::{memory_management::MemoryManagement, storage::BytesStorage};

use crate::compiler::{builtin::BuiltinArray, memref::LineMemRef};

use super::passes::shared_memories::SharedMemories;

struct SharedMlirData {
    args_zero_indirection: Vec<LineMemRef>,
    metadata: Vec<u32>,
    args_first_indirection: Vec<*mut ()>,
    scalars: Vec<ScalarBinding>,
}

unsafe impl Send for SharedMlirData {}
unsafe impl Sync for SharedMlirData {}

pub struct MlirData {
    shared_mlir_data: Arc<SharedMlirData>,
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
        bindings: Bindings,
        shared_memories: &SharedMemories,
        memory_management: &mut MemoryManagement<BytesStorage>,
    ) -> Self {
        let Bindings {
            buffers,
            scalars,
            metadata,
            ..
        } = bindings;

        let scalars_binding: Vec<_> = scalars.into_values().collect();

        let builtin = BuiltinArray::default();
        let max_buffer_size = buffers.len() + scalars_binding.len() + BuiltinArray::len();

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

        for b in buffers {
            let handle = memory_management
                .get_resource(b.memory, b.offset_start, b.offset_end)
                .expect("Failed to find resource");
            let ptr = handle.write();
            let line_memref = LineMemRef::new(ptr);
            push_undirected(line_memref);
        }

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
        for shared_memory in shared_memories.0.iter() {
            let length = (shared_memory.elem.size() * shared_memory.length as usize) as u64;
            let handle = memory_management.reserve(length, None);
            let b = Handle::new(handle, None, None, length).binding();
            let handle = memory_management
                .get_resource(b.memory, b.offset_start, b.offset_end)
                .expect("Failed to find resource");
            let ptr = handle.write();
            let line_memref = LineMemRef::new(ptr);
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
