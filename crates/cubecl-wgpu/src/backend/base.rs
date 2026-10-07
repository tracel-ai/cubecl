use super::wgsl;
use crate::AutoRepresentationRef;
use cubecl_core::WgpuCompilationOptions;
use cubecl_ir::{DeviceProperties, PhysicalDevice};
use cubecl_server::compiler::CompilationError;
use wgpu::{Adapter, Device, Queue};

#[cfg(feature = "spirv")]
use super::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use super::metal;

#[cfg(windows)]
use super::dx12;

#[cfg(target_vendor = "apple")]
use super::metal_card;

/// What a shader module is built from: the compiler's representation and the
/// source text, reconciled.
///
/// A compiled kernel carries both, and the representation says which the
/// device wants. A precompiled kernel carries only text, so the language it
/// was tagged with decides instead.
pub enum ModuleSource<'a> {
    /// An assembled SPIR-V module, handed to the driver as is.
    #[cfg(feature = "spirv")]
    SpirV(&'a cubecl_spirv::SpirvKernel),
    /// Metal Shading Language text, handed to the driver as is.
    #[cfg(all(feature = "msl", target_os = "macos"))]
    Msl(&'a str),
    /// WGSL text, for naga to compile.
    Wgsl(&'a str),
}

impl<'a> ModuleSource<'a> {
    /// Pairs `repr` with `source`, or, when there is no representation, reads
    /// the language off `lang`, the tag the precompiled kernel was accepted
    /// under.
    pub fn resolve(
        repr: Option<AutoRepresentationRef<'a>>,
        lang: &str,
        source: &'a str,
    ) -> Result<Self, CompilationError> {
        match repr {
            #[cfg(feature = "spirv")]
            Some(AutoRepresentationRef::SpirV(repr)) => Ok(Self::SpirV(repr)),
            Some(AutoRepresentationRef::Wgsl(_)) => Ok(Self::Wgsl(source)),
            #[cfg(feature = "msl")]
            Some(AutoRepresentationRef::Msl(_)) => Self::msl(source),
            None => match lang {
                "wgsl" => Ok(Self::Wgsl(source)),
                "msl" => Self::msl(source),
                other => Err(CompilationError::Generic {
                    reason: format!(
                        "wgpu has no text passthrough for a precompiled `{other}` kernel"
                    ),
                    backtrace: cubecl_environment::backtrace::BackTrace::capture(),
                }),
            },
        }
    }

    #[cfg(all(feature = "msl", target_os = "macos"))]
    fn msl(source: &'a str) -> Result<Self, CompilationError> {
        Ok(Self::Msl(source))
    }

    #[cfg(not(all(feature = "msl", target_os = "macos")))]
    fn msl(_source: &'a str) -> Result<Self, CompilationError> {
        Err(CompilationError::Generic {
            reason: "MSL passthrough is only available on macOS".to_string(),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        })
    }
}

/// Request a device, returning device creation failures.
pub async fn try_request_device(
    adapter: &Adapter,
) -> Result<(Device, Queue), crate::WgpuInitError> {
    if let Some(result) = request_vulkan_device(adapter).await? {
        return Ok(result);
    }
    if let Some(result) = request_metal_device(adapter).await? {
        return Ok(result);
    }
    wgsl::try_request_device(adapter).await
}

#[cfg(feature = "spirv")]
async fn request_vulkan_device(
    adapter: &Adapter,
) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    if is_vulkan(adapter) {
        vulkan::try_request_vulkan_device(adapter).await
    } else {
        Ok(None)
    }
}

#[cfg(not(feature = "spirv"))]
async fn request_vulkan_device(
    _adapter: &Adapter,
) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    Ok(None)
}

#[cfg(all(feature = "msl", target_os = "macos"))]
async fn request_metal_device(
    adapter: &Adapter,
) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    if is_metal(adapter) {
        metal::try_request_metal_device(adapter).await.map(Some)
    } else {
        Ok(None)
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
async fn request_metal_device(
    _adapter: &Adapter,
) -> Result<Option<(Device, Queue)>, crate::WgpuInitError> {
    Ok(None)
}

pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    if register_vulkan_features(adapter, props, comp_options) {
        return;
    }
    if register_metal_features(adapter, props, comp_options) {
        return;
    }
    wgsl::register_wgsl_features(adapter, props, comp_options);
}

#[cfg(feature = "spirv")]
pub fn register_vulkan_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) -> bool {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options)
    } else {
        false
    }
}

#[cfg(not(feature = "spirv"))]
pub fn register_vulkan_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
) -> bool {
    false
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub fn register_metal_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) -> bool {
    if is_metal(adapter) {
        metal::register_metal_features(adapter, props, comp_options)
    } else {
        false
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
pub fn register_metal_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
) -> bool {
    false
}

/// The card behind `adapter`, `None` for a software adapter, which is no card at all.
#[cfg_attr(
    not(any(feature = "spirv", windows, target_vendor = "apple")),
    expect(unused_variables)
)]
pub fn physical_device(adapter: &Adapter, info: &wgpu::AdapterInfo) -> Option<PhysicalDevice> {
    if info.device_type == wgpu::DeviceType::Cpu {
        return None;
    }
    let mut physical = PhysicalDevice::default();
    // Metal and WebGPU report a zero vendor rather than none.
    physical.vendor = (info.vendor != 0).then(|| info.vendor.into());
    // wgpu gives a DX12 adapter the address of the first card with its vendor and device id, so
    // identical cards would share one.
    if info.backend == wgpu::Backend::Vulkan {
        physical.pci_address = info.device_pci_bus_id.parse().ok();
    }
    #[cfg(feature = "spirv")]
    if is_vulkan(adapter) {
        vulkan::describe_card(adapter, &mut physical);
    }
    #[cfg(windows)]
    dx12::describe_card(adapter, &mut physical);
    #[cfg(target_vendor = "apple")]
    metal_card::describe_card(adapter, &mut physical);
    Some(physical)
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>().is_some() }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
fn is_metal(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Metal>().is_some() }
}
