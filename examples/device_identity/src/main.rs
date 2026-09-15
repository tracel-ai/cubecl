//! Prints who each device is, through every runtime this build reaches it with. On a machine
//! whose cards are visible to both CUDA and Vulkan, the same card must report the same PCI
//! address and UUID both ways.

use cubecl::ir::DeviceIdentity;
use cubecl_runtime::runtime::Runtime;

fn main() {
    #[cfg(feature = "cuda")]
    {
        use cubecl::cuda::{CudaDevice, CudaRuntime};

        let count = cudarc::driver::result::device::get_count().unwrap_or(0) as usize;
        for index in 0..count {
            let client = CudaRuntime::client(&CudaDevice { index });
            report(&format!("cuda:{index}"), &client.properties().identity);
        }
    }

    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind, WgpuRuntime};

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = cubecl_environment::future::block_on(
            instance.enumerate_adapters(wgpu::Backends::VULKAN),
        );
        let discrete = adapters
            .iter()
            .filter(|adapter| adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .count();
        for index in 0..discrete {
            let device = WgpuDevice {
                kind: WgpuDeviceKind::DiscreteGpu(index),
                backend: WgpuBackend::Vulkan,
            };
            let client = WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(&device);
            report(&format!("vulkan:{index}"), &client.properties().identity);
        }
    }
}

fn report(device: &str, identity: &DeviceIdentity) {
    println!("{device}: {} ({})", identity.name, identity.fingerprint);
    let Some(physical) = &identity.physical else {
        println!("  no card");
        return;
    };
    let pci = physical
        .pci
        .map(|pci| pci.to_string())
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
    println!(
        "  pci {pci}  uuid {uuid}  vendor {:?}  device {:?}  memory {memory}",
        physical.vendor_id.map(|id| format!("{id:#06x}")),
        physical.device_id.map(|id| format!("{id:#06x}")),
    );
}
