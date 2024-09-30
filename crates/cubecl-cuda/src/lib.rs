#[macro_use]
extern crate derive_new;
extern crate alloc;

mod compute;
mod device;
mod runtime;

pub mod compiler;
pub use device::*;

pub use runtime::CudaRuntime;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::CudaRuntime;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_cmma_old!();
    cubecl_linalg::testgen_tiling2d_old!();
    cubecl_linalg::testgen_matmul_instruction_cmma_f16!();
}
