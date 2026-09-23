//! Aliasing and access attributes on a GPU kernel's buffer parameters.

use crate::{prelude::BufferIOAttr, shared::llvm_module::EntryFunction};

/// Marks every pointer parameter `noalias`, and the read-only buffers and the metadata
/// `readonly`, from the recorded access modes. Distinct bindings never overlap, which is
/// what lets the backend keep a value in a register across a store, and a read-only pointer is
/// what lets NVPTX load through the non-coherent cache and AMDGPU prove a uniform load is not
/// clobbered and make it a scalar load.
///
/// `entry`'s parameters are the buffers in binding order followed by `metadata_params`
/// metadata parameters. An LLVM without one of the two attributes gets the other alone.
pub(crate) fn annotate_buffer_params(
    entry: &EntryFunction<'_>,
    io: &[BufferIOAttr],
    metadata_params: u32,
) {
    // Atomic loads must retain coherent memory access.
    let may_say_readonly = !entry.instructions().any(|inst| inst.is_atomic_load());

    let (buffers, metadata) = entry.split_params(metadata_params);
    let buffers = buffers.enumerate().map(|(binding, param)| {
        let read_only = io
            .get(binding)
            .is_some_and(|attr| *attr == BufferIOAttr::ReadOnly);
        (param, read_only)
    });
    let metadata = metadata.map(|param| (param, true));

    for (param, read_only) in buffers.chain(metadata) {
        if !entry.param_is_pointer(param) {
            continue;
        }
        let _ = entry.add_param_attribute(param, "noalias", 0);
        if read_only && may_say_readonly {
            let _ = entry.add_param_attribute(param, "readonly", 0);
        }
    }
}
