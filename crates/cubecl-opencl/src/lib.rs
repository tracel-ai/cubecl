//! `OpenCL` runtime for `CubeCL`: kernels in `OpenCL` C 3.0 (`cubecl-cpp`'s `OpenCL` dialect) built by the device's
//! driver, buffers in Intel's USM device allocations where the device has them, else in coarse-grained shared
//! virtual memory (either way a slice of a page is a pointer the kernel takes as it is), a command queue per
//! stream. Planes run on subgroups where the device fixes their size (`cl_intel_required_subgroup_size`), and
//! cooperative matrices on Intel's matrix engines (DPAS, through `cl_intel_subgroup_matrix_multiply_accumulate`).
//!
//! A device takes part if it has coarse-grained shared virtual memory and `OpenCL` C 3.0.

mod compute;
mod device;
mod memory;
mod runtime;

pub use compute::OpenClServer;
pub use device::OpenClDevice;
pub use memory::{OpenClResource, OpenClStorage};
pub use runtime::OpenClRuntime;

pub(crate) use cubecl_cpp::opencl::OpenClCompiler;

#[cfg(test)]
mod tests_dialect;
#[cfg(test)]
mod tests_memory;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::OpenClRuntime;

    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);
    cubecl_core::testgen_profiling!();
}
