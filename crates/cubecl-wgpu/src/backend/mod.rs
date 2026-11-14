mod base;
pub(crate) mod wgsl;

#[cfg(feature = "spirv")]
pub mod vulkan;

#[cfg(any(
    feature = "spirv",
    all(not(target_family = "wasm"), not(target_os = "macos"))
))]
pub(crate) mod vulkan_memory;

/// No Vulkan driver to ask on this target: the device has no figure.
#[cfg(not(any(
    feature = "spirv",
    all(not(target_family = "wasm"), not(target_os = "macos"))
)))]
pub(crate) mod vulkan_memory {
    pub(crate) fn memory_available(
        _: &wgpu::Device,
        _: wgpu::BufferUsages,
        _: bool,
    ) -> Option<u64> {
        None
    }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub mod metal;

#[cfg(windows)]
pub mod dx12;

#[cfg(target_vendor = "apple")]
pub mod metal_card;

pub use base::*;
