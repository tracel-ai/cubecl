// We cannot put this struct in cubecl-wgpu crate due to circular dependencies.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_u64: bool,
    /// Whether the Vulkan compiler is supported or we need to fall back to WGSL
    pub supports_vulkan_compiler: bool,
    pub supports_msl_compiler: bool,

    pub vulkan: VulkanCompilationOptions,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VulkanCompilationOptions {
    pub supports_fp_fast_math: bool,
    pub supports_explicit_smem: bool,
    pub supports_long_vectors: bool,
    pub supports_arbitrary_bitwise: bool,
    pub supports_uniform_standard_layout: bool,
    pub supports_uniform_unsized_array: bool,
    pub supports_float8: bool,
    pub supports_dp4a: bool,
    pub supports_signed_zero_inf_nan_preserve_f16: bool,
    pub supports_signed_zero_inf_nan_preserve_f32: bool,
    pub supports_signed_zero_inf_nan_preserve_f64: bool,

    pub max_spirv_version: (u8, u8),
    pub max_vector_size: usize,
    pub push_constant_size: usize,
}

impl VulkanCompilationOptions {
    /// Whether the device keeps NaN, infinity and the sign of zero for floats of `width` bits when
    /// asked to.
    pub fn preserves_special_values(&self, width: u32) -> bool {
        match width {
            16 => self.supports_signed_zero_inf_nan_preserve_f16,
            32 => self.supports_signed_zero_inf_nan_preserve_f32,
            64 => self.supports_signed_zero_inf_nan_preserve_f64,
            _ => false,
        }
    }
}
