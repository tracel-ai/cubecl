/// Which of the two WGSL front ends reads the module. They disagree on the
/// subgroup builtins: Tint takes them only under `enable subgroups;` and
/// refuses a call from control flow it cannot prove uniform unless that
/// diagnostic is off, while Naga takes them without a directive and rejects
/// the directive itself. Natively wgpu is Naga; in a browser it is whichever
/// the browser embeds, Tint in Chromium and Naga in Firefox.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WgslFrontEnd {
    #[default]
    Naga,
    Tint,
}

// We cannot put this struct in cubecl-wgpu crate due to circular dependencies.
#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_u64: bool,
    /// Whether the Vulkan compiler is supported or we need to fall back to WGSL
    pub supports_vulkan_compiler: bool,
    pub supports_msl_compiler: bool,
    /// The plane width a kernel that uses plane operations is pinned to, on a
    /// device whose planes would otherwise vary from kernel to kernel (the
    /// browser's WebGPU, where the native pinning is out of reach). `None`
    /// where the width is fixed anyway.
    pub pinned_plane_size: Option<u32>,
    /// The WGSL front end the module is written for.
    pub wgsl_front_end: WgslFrontEnd,

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

    pub max_spirv_version: (u8, u8),
    pub max_vector_size: usize,
    pub push_constant_size: usize,
}
