pub mod cuda_compiler;
pub use cuda_compiler::*;
pub mod ptx_wmma_compiler;
pub use ptx_wmma_compiler::*;

const WMMA_NAMESPACE: &str = "nvcuda::wmma";
const WMMA_MINIMUM_VERSION: u32 = 70;
