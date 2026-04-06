// We cannot put this struct in cubecl-wgpu crate due to circular dependencies.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_u64: bool,
    /// Whether the Vulkan compiler is supported or we need to fall back to WGSL
    pub supports_vulkan_compiler: bool,

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

    pub max_spirv_version: (u8, u8),
    pub max_vector_size: usize,
    pub push_constant_size: usize,
}
