use std::sync::Arc;

use cubecl_core::server::ScalarBinding;
use cubecl_runtime::storage::BytesResource;

use crate::compiler::{builtin::Builtin, memref::LineMemRef};

struct SharedMlirData {
    args_zero_indirection: Vec<LineMemRef>,
    args_first_indirection: Vec<*mut ()>,
    scalars: Vec<ScalarBinding>,
}

pub struct MlirData {
    shared_mlir_data: Arc<SharedMlirData>,
    pub args_second_indirection: Vec<*mut ()>,
    pub builtin: Builtin,
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
    pub fn new(handles: Vec<BytesResource>, scalars_binding: Vec<ScalarBinding>) -> Self {
        let builtin = Builtin::default();
        let max_buffer_size = handles.len() + scalars_binding.len() + builtin.len();

        let args_zero_indirection = Vec::with_capacity(max_buffer_size);
        let args_first_indirection = Vec::with_capacity(max_buffer_size);
        let mut args_second_indirection = Vec::with_capacity(max_buffer_size);
        let scalars: Vec<ScalarBinding> = Vec::with_capacity(max_buffer_size);

        let mut shared_mlir_data = SharedMlirData {
            args_zero_indirection,
            args_first_indirection,
            scalars,
        };

        for handle in handles {
            let ptr = handle.write();
            let first_box = LineMemRef::new(ptr);
            shared_mlir_data.args_zero_indirection.push(first_box);
            let undirected =
                shared_mlir_data.args_zero_indirection.last_mut().unwrap() as *mut LineMemRef;
            shared_mlir_data
                .args_first_indirection
                .push(undirected as *mut ());
            let undirected =
                shared_mlir_data.args_first_indirection.last_mut().unwrap() as *mut *mut ();

            args_second_indirection.push(undirected as *mut ());
        }

        for scalar in scalars_binding.into_iter() {
            shared_mlir_data.scalars.push(scalar);
            let data = shared_mlir_data
                .scalars
                .last_mut()
                .unwrap()
                .data
                .as_mut_ptr() as *mut u8;
            args_second_indirection.push(data as *mut ());
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
                .push(arg as *mut u64 as *mut ());
        }
    }
}
