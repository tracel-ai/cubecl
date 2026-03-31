// We cannot put this struct in cubecl-wgpu crate due to circular dependencies.
#[derive(Clone, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_fp_fast_math: bool,
    pub supports_u64: bool,
    pub supports_explicit_smem: bool,
    pub supports_arbitrary_bitwise: bool,
    pub supports_uniform_standard_layout: bool,
    pub supports_uniform_unsized_array: bool,
}
