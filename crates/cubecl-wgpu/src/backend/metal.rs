use cubecl_core::compute::Visibility;
use cubecl_msl::MetalKernel;

pub fn bindings(repr: &MetalKernel) -> Vec<(usize, Visibility)> {
    let mut bindings: Vec<(usize, Visibility)> = vec![];
    // must be in the same order as the compilation order: inputs, outputs and named
    let mut buffer_idx = 0;
    for b in repr.inputs.iter() {
        bindings.push((buffer_idx, b.address_space.into()));
        buffer_idx += 1;
    }
    for b in repr.outputs.iter() {
        bindings.push((buffer_idx, b.address_space.into()));
        buffer_idx += 1;
    }
    for (_name, b) in repr.named.iter() {
        bindings.push((buffer_idx, b.address_space.into()));
        buffer_idx += 1;
    }
    bindings
}
