use std::ffi::{c_longlong, c_void};

#[repr(C)]
pub struct MemRef<const N_DIMS: usize> {
    /// Pointer to the allocated memory
    allocated: *mut c_void,
    /// Aligned pointer to the allocated memory in our case almost always the same as the allocated pointer
    aligned: *mut c_void,
    /// Offset from the start that is almost always zero for CubeCL CPU allocator
    offset: c_longlong,
    /// Shape of the memory
    shape: [c_longlong; N_DIMS],
    /// Stride in elements of the memory for each dimension
    stride: [c_longlong; N_DIMS],
}

// For the moment, the memory is only considered as a simple contiguous memory array in MLIR, but maybe we could improve access pattern by adding dimensions
type LineMemRef = MemRef<1>;

impl LineMemRef {
    pub fn new() {}
}
