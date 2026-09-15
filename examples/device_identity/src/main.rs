//! Prints who each device is, through every runtime this build reaches it with. On a machine
//! whose cards are visible to both CUDA and Vulkan, the same card must report the same PCI
//! address and UUID both ways; on Windows, DirectX 12 and Vulkan must report the same LUID.

use cubecl::ir::DeviceIdentity;
use cubecl_runtime::runtime::Runtime;

fn main() {
    #[cfg(feature = "cuda")]
    {
        use cubecl::cuda::{CudaDevice, CudaRuntime};

        cudarc::driver::result::init().expect("the CUDA driver loads");
        let count = cudarc::driver::result::device::get_count().unwrap_or(0) as usize;
        for index in 0..count {
            let client = CudaRuntime::client(&CudaDevice { index });
            report(&format!("cuda:{index}"), &client.properties().identity);
        }
    }

    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind, WgpuRuntime};

        let apis = [
            (wgpu::Backends::VULKAN, WgpuBackend::Vulkan, "vulkan"),
            (wgpu::Backends::DX12, WgpuBackend::Dx12, "dx12"),
        ];
        for (backends, backend, api) in apis {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapters =
                cubecl_environment::future::block_on(instance.enumerate_adapters(backends));
            let count = |kind| {
                adapters
                    .iter()
                    .filter(|adapter| adapter.get_info().device_type == kind)
                    .count()
            };
            let discrete =
                (0..count(wgpu::DeviceType::DiscreteGpu)).map(WgpuDeviceKind::DiscreteGpu);
            let integrated =
                (0..count(wgpu::DeviceType::IntegratedGpu)).map(WgpuDeviceKind::IntegratedGpu);
            for kind in discrete.chain(integrated) {
                let label = format!("{api}:{kind:?}");
                let client = WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(&WgpuDevice {
                    kind,
                    backend,
                });
                report(&label, &client.properties().identity);
            }
        }
    }
}

fn report(device: &str, identity: &DeviceIdentity) {
    println!("{device}: {} ({})", identity.name, identity.fingerprint);
    let Some(physical) = &identity.physical else {
        println!("  no card");
        return;
    };
    let pci_address = physical
        .pci_address
        .map(|id| id.to_string())
        .unwrap_or_else(|| "?".into());
    let uuid = physical
        .uuid
        .map(|uuid| {
            uuid.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .unwrap_or_else(|| "?".into());
    let memory = physical
        .total_memory
        .map(|bytes| format!("{} MiB", bytes >> 20))
        .unwrap_or_else(|| "?".into());
    let luid = physical
        .luid
        .map(|luid| {
            luid.bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .unwrap_or_else(|| "?".into());
    println!(
        "  pci {pci_address}  uuid {uuid}  luid {luid}  vendor {}  device {:?}  memory {memory}",
        physical
            .vendor
            .map(|vendor| vendor.to_string())
            .unwrap_or_else(|| "?".into()),
        physical.device_id.map(|id| format!("{id:#06x}")),
    );
}
