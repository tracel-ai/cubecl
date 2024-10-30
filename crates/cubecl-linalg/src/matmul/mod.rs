/// Contains cmma_old kernel
pub mod cmma_old;
/// Matrix multiplication with components.
pub mod matmul_modular;
/// Tests for matmul kernels
pub mod tests;
/// Contains tiling2d kernel
pub mod tiling2d;

mod base;
pub use base::*;
