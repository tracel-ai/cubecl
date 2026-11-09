//! The entry point's tensor map parameters.
//!
//! A tensor map is a binding, but the runtime launches it by value: a 128-byte `CUtensorMap` in
//! the parameter space, not a pointer to a buffer.

use crate::{prelude::*, shared::llvm_module::EntryFunction};
use cubecl_core::ir::attributes::{
    ATTR_BUFFER_BINDING, ATTR_TENSOR_MAP_BINDING, BufferBindingAttr, FuncInterface,
};

/// A `CUtensorMap` is 128 bytes, aligned to 64.
const TENSOR_MAP_BYTES: u64 = 128;
const TENSOR_MAP_ALIGN: u64 = 64;

/// The entry point's parameters that are tensor maps, by binding position.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorMapParams(Vec<usize>);

impl TensorMapParams {
    /// The tensor map arguments of `entry_func`.
    pub(crate) fn new(ctx: &Context, entry_func: FuncOp) -> Self {
        let args = entry_func
            .get_entry_block(ctx)
            .deref(ctx)
            .get_num_arguments();
        let bindings = (0..args)
            .filter(|&arg| entry_func.has_arg_attr(ctx, arg, &ATTR_TENSOR_MAP_BINDING))
            .filter_map(|arg| {
                entry_func
                    .get_arg_attr::<BufferBindingAttr>(ctx, arg, &ATTR_BUFFER_BINDING)
                    .map(|binding| binding.buffer_pos)
            })
            .collect();
        Self(bindings)
    }

    /// The binding positions of the tensor maps.
    pub(crate) fn bindings(&self) -> &[usize] {
        &self.0
    }

    /// Passes each tensor map by value, as the runtime launches it, and declares it a grid
    /// constant: a TMA instruction reads the map through its address, and without the promise
    /// LLVM would copy the parameter to local memory first, where TMA cannot read it.
    pub(crate) fn mark(&self, entry: &EntryFunction<'_>) {
        for &binding in &self.0 {
            let param = entry.param(binding as u32);
            let byval = entry.add_param_byval(param, TENSOR_MAP_BYTES);
            let aligned = entry.add_param_attribute(param, "align", TENSOR_MAP_ALIGN);
            assert!(
                byval && aligned,
                "this LLVM has no `byval` or `align` attribute, so the tensor map parameter \
                 cannot be declared"
            );
            entry.add_param_string_attribute(param, "nvvm.grid_constant", "");
        }
    }
}
