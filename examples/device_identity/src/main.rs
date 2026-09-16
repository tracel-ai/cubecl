//! Prints who each device is, through every runtime this build reaches it with, then which
//! devices are one card. A card visible to CUDA and Vulkan must report one PCI address both ways;
//! on Windows, DirectX 12 and Vulkan must report one LUID.

#[cfg(feature = "cuda")]
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::ir::{DeviceIdentity, PhysicalDevice};
#[cfg(feature = "wgpu")]
use cubecl::wgpu::{AutoCompiler, WgpuBackend, WgpuDevice, WgpuDeviceKind, WgpuRuntime};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use cubecl_runtime::runtime::Runtime;

fn main() {
    #[cfg_attr(not(any(feature = "cuda", feature = "wgpu")), expect(unused_mut))]
    let mut devices: Vec<(String, DeviceIdentity)> = Vec::new();

    #[cfg(feature = "cuda")]
    if cudarc::driver::result::init().is_ok() {
        let count = cudarc::driver::result::device::get_count().unwrap_or(0) as usize;
        for index in 0..count {
            let client = CudaRuntime::client(&CudaDevice { index });
            devices.push((
                format!("cuda:{index}"),
                client.properties().identity.clone(),
            ));
        }
    }

    #[cfg(feature = "wgpu")]
    for (backends, backend, api) in [
        (wgpu::Backends::VULKAN, WgpuBackend::Vulkan, "vulkan"),
        (wgpu::Backends::DX12, WgpuBackend::Dx12, "dx12"),
    ] {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = cubecl_environment::future::block_on(instance.enumerate_adapters(backends));
        let count = |kind| {
            adapters
                .iter()
                .filter(|adapter| adapter.get_info().device_type == kind)
                .count()
        };
        let kinds = (0..count(wgpu::DeviceType::DiscreteGpu))
            .map(WgpuDeviceKind::DiscreteGpu)
            .chain((0..count(wgpu::DeviceType::IntegratedGpu)).map(WgpuDeviceKind::IntegratedGpu))
            .chain((0..count(wgpu::DeviceType::VirtualGpu)).map(WgpuDeviceKind::VirtualGpu));
        for kind in kinds {
            let label = format!("{api}:{kind:?}");
            let client = WgpuRuntime::<AutoCompiler>::client(&WgpuDevice { kind, backend });
            devices.push((label, client.properties().identity.clone()));
        }
    }

    for (label, identity) in &devices {
        report(label, identity);
    }

    let mut cards: Vec<(&PhysicalDevice, Vec<&str>)> = Vec::new();
    for (label, identity) in &devices {
        let Some(physical) = &identity.physical else {
            continue;
        };
        match cards
            .iter_mut()
            .find(|(card, _)| card.is_same_card(physical))
        {
            Some((_, labels)) => labels.push(label),
            None => cards.push((physical, vec![label])),
        }
    }
    for (_, labels) in cards {
        println!("one card: {}", labels.join(", "));
    }
}

fn report(device: &str, identity: &DeviceIdentity) {
    println!("{device}: {} ({})", identity.name, identity.fingerprint);
    let Some(physical) = &identity.physical else {
        println!("  no card");
        return;
    };
    let unknown = || "?".to_string();
    let pci_address = physical
        .pci_address
        .map_or_else(unknown, |address| address.to_string());
    let luid = physical.luid.map_or_else(unknown, |luid| {
        luid.bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    });
    let vendor = physical
        .vendor
        .map_or_else(unknown, |vendor| vendor.to_string());
    println!("  pci {pci_address}  luid {luid}  vendor {vendor}");
}
