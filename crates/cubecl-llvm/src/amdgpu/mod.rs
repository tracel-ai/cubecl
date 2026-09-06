//! The AMDGPU (`amdgcn-amd-amdhsa`) target.

pub mod abi;
pub mod builtins;
pub mod codegen;
pub mod device_libs;
pub mod intrinsic;
pub mod lld;
pub mod matrix;
pub mod ocml;
pub mod plane;
pub mod plane_reduce;
pub mod printf;
pub mod shared_memory;
pub mod synchronization;

pub(crate) const AMDGPU_DISABLED: &str =
    "AMDGPU code generation requires the `cubecl-llvm/amdgpu` feature";

/// Shared by the compiler and direct code-generation/linking entry points.
pub(crate) fn require_amdgpu() -> Result<(), String> {
    if cfg!(feature = "amdgpu") {
        Ok(())
    } else {
        Err(AMDGPU_DISABLED.into())
    }
}
