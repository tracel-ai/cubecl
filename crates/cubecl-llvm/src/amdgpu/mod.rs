//! The AMDGPU (`amdgcn-amd-amdhsa`) target.

pub mod abi;
pub mod builtins;
pub mod codegen;
pub mod device_libs;
pub mod lld;
pub mod ocml;
pub mod plane;
pub mod plane_reduce;
pub mod printf;
pub mod shared_memory;
pub mod synchronization;

/// Wavefront width of `arch`: 32 on RDNA (gfx10 and later), 64 on GCN and CDNA.
pub fn plane_dim_for(arch: &str) -> u32 {
    let rdna = arch.starts_with("gfx10") || arch.starts_with("gfx11") || arch.starts_with("gfx12");
    if rdna { 32 } else { 64 }
}
