//! The AMDGPU (`amdgcn-amd-amdhsa`) target. See
//! `docs/superpowers/specs/2026-08-24-cubecl-llvm-amdgpu-design.md`.

pub mod abi;
pub mod builtins;

/// Wavefront width of `arch`: 32 on RDNA (gfx10 and later), 64 on GCN and CDNA.
pub fn plane_dim_for(arch: &str) -> u32 {
    let rdna = arch.starts_with("gfx10") || arch.starts_with("gfx11") || arch.starts_with("gfx12");
    if rdna { 32 } else { 64 }
}
