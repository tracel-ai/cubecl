mod base;
pub(crate) mod wgsl;

#[cfg(feature = "spirv")]
pub mod vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub mod metal;

#[cfg(windows)]
pub mod dx12;

#[cfg(target_vendor = "apple")]
pub mod metal_card;

pub use base::*;
