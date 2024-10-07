extern crate alloc;

use std::sync::Arc;

use cubecl_core::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    ir::{Elem, FloatKind},
    Feature, Runtime,
};
use cubecl_runtime::{
    memory_management::{dynamic::DynamicMemoryManagement, MemoryDeviceProperties},
    DeviceProperties, ComputeRuntime,
};
use cubecl_spirv::{GLCompute, SpirvCompiler};
use cubecl_wgpu::{
    create_wgpu_setup, init_async, init_memory_management, AutoGraphicsApi, RuntimeOptions, Vulkan,
    WgpuDevice, WgpuStorage,
};
use server::WgpuSpirvServer;
use wgpu::hal;

mod server;

/// Runtime that uses the [wgpu] crate with the SPIR-V compiler.
#[derive(Debug)]
pub struct WgpuSpirvRuntime;

#[cfg(not(simple_memory_management))]
type MemoryManagement = DynamicMemoryManagement<WgpuStorage>;
#[cfg(simple_memory_management)]
type MemoryManagement = SimpleMemoryManagement<WgpuStorage>;

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

type Server = WgpuSpirvServer<MemoryManagement>;

impl Runtime for WgpuSpirvRuntime {
    type Compiler = SpirvCompiler<GLCompute>;
    type Server = WgpuSpirvServer<MemoryManagement>;

    type Channel = MutexComputeChannel<WgpuSpirvServer<MemoryManagement>>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            let (adapter, device_wgpu, queue) =
                pollster::block_on(create_wgpu_setup::<AutoGraphicsApi>(device));
            create_client(adapter, device_wgpu, queue, RuntimeOptions::default())
        })
    }

    fn name() -> &'static str {
        "wgpu"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[4, 2]
    }
}

/// Initialize a client on the given device with the given options. This function is useful to configure the runtime options
/// or to pick a different graphics API. On wasm, it is necessary to use [`init_async`] instead.
pub fn init_sync(device: &WgpuDevice, options: RuntimeOptions) {
    pollster::block_on(init_async::<Vulkan>(device, options));
}

pub fn create_client(
    adapter: Arc<wgpu::Adapter>,
    device_wgpu: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    options: RuntimeOptions,
) -> ComputeClient<
    WgpuSpirvServer<MemoryManagement>,
    MutexComputeChannel<WgpuSpirvServer<MemoryManagement>>,
> {
    let limits = device_wgpu.limits();
    let mem_props = MemoryDeviceProperties {
        max_page_size: limits.max_storage_buffer_binding_size as usize,
        alignment: limits.min_storage_buffer_offset_alignment as usize,
    };

    let memory_management = init_memory_management(
        device_wgpu.clone(),
        mem_props.clone(),
        options.memory_config,
    );
    let server = WgpuSpirvServer::new(memory_management, device_wgpu, queue, options.tasks_max);
    let channel = MutexComputeChannel::new(server);

    let features = adapter.features();
    let has_cmma = unsafe {
        adapter.as_hal::<hal::api::Vulkan, _, _>(|adapter| {
            if let Some(adapter) = adapter {
                adapter
                    .physical_device_capabilities()
                    .supports_extension(c"VK_KHR_cooperative_matrix")
            } else {
                false
            }
        })
    };
    let mut device_props = DeviceProperties::new(&[], mem_props);

    if features.contains(wgpu::Features::SUBGROUP) {
        device_props.register_feature(Feature::Subcube);
    }
    if has_cmma {
        register_cmma_features(&mut device_props);
    }

    ComputeClient::new(channel, device_props)
}

fn register_cmma_features(properties: &mut DeviceProperties<Feature>) {
    // Types fully supported.
    for (a, b, c) in [
        (
            Elem::Float(FloatKind::F16),
            Elem::Float(FloatKind::F16),
            Elem::Float(FloatKind::F16),
        ),
        (
            Elem::Float(FloatKind::F16),
            Elem::Float(FloatKind::F16),
            Elem::Float(FloatKind::F32),
        ),
        (
            Elem::Float(FloatKind::BF16),
            Elem::Float(FloatKind::BF16),
            Elem::Float(FloatKind::F32),
        ),
    ] {
        properties.register_feature(Feature::Cmma {
            a,
            b,
            c,
            m: 16,
            k: 16,
            n: 16,
        });
        properties.register_feature(Feature::Cmma {
            a,
            b,
            c,
            m: 32,
            k: 16,
            n: 8,
        });
        properties.register_feature(Feature::Cmma {
            a,
            b,
            c,
            m: 8,
            k: 16,
            n: 32,
        });
    }
}

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::WgpuSpirvRuntime;

    cubecl_core::testgen_all!();
    cubecl_linalg::testgen_all!();
}
