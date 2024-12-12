#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub use device::*;
pub use runtime::*;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;
    pub use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, bf16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_linalg::testgen_cmma_matmul!([f16]);
    cubecl_linalg::testgen_plane_mma!([f16], f16);
    cubecl_linalg::testgen_plane_mma!([f16], f32);
    cubecl_linalg::testgen_tiling2d!([f16, bf16, f32]);
    cubecl_linalg::testgen_cmma_old!([f16, bf16, f32 /*, f64*/]);
    cubecl_reduce::testgen_reduce!([f16, bf16, f32, f64]);
}
