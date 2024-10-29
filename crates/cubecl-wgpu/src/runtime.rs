use std::marker::PhantomData;

use crate::{
    compiler::{base::WgpuCompiler, wgsl::WgslCompiler},
    compute::{WgpuServer, WgpuStorage},
    AutoGraphicsApi, GraphicsApi, WgpuDevice,
};
use alloc::sync::Arc;
use cubecl_common::future;
use cubecl_core::{Feature, Runtime};
pub use cubecl_runtime::memory_management::MemoryConfiguration;
use cubecl_runtime::DeviceProperties;
use cubecl_runtime::{channel::MutexComputeChannel, client::ComputeClient, ComputeRuntime};
use cubecl_runtime::{
    memory_management::{MemoryDeviceProperties, MemoryManagement},
    storage::ComputeStorage,
};

/// Runtime that uses the [wgpu] crate with the wgsl compiler. This is used in the Wgpu backend.
/// For advanced configuration, use [`init_sync`] to pass in runtime options or to select a
/// specific graphics API.
#[derive(Debug)]
pub struct WgpuRuntime<C: WgpuCompiler = WgslCompiler>(PhantomData<C>);

type Server = WgpuServer<WgslCompiler>;

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

impl Runtime for WgpuRuntime<WgslCompiler> {
    type Compiler = WgslCompiler;
    type Server = WgpuServer<WgslCompiler>;

    type Channel = MutexComputeChannel<WgpuServer<WgslCompiler>>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            let setup = future::block_on(create_setup_for_device::<AutoGraphicsApi, WgslCompiler>(
                device,
            ));
            create_client_on_setup(setup, RuntimeOptions::default())
        })
    }

    fn name() -> &'static str {
        "wgpu<wgsl>"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[4, 2]
    }
}

/// The values that control how a WGPU Runtime will perform its calculations.
pub struct RuntimeOptions {
    /// Control the amount of compute tasks to be aggregated into a single GPU command.
    pub tasks_max: usize,
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        #[cfg(test)]
        const DEFAULT_MAX_TASKS: usize = 1;
        #[cfg(not(test))]
        const DEFAULT_MAX_TASKS: usize = 16;

        let tasks_max = match std::env::var("CUBECL_WGPU_MAX_TASKS") {
            Ok(value) => value
                .parse::<usize>()
                .expect("CUBECL_WGPU_MAX_TASKS should be a positive integer."),
            Err(_) => DEFAULT_MAX_TASKS,
        };

        Self {
            tasks_max,
            memory_config: MemoryConfiguration::default(),
        }
    }
}

/// A complete setup used to run wgpu on a GPU.
///
/// These can either be created with [`ini
#[derive(Clone)]
pub struct WgpuSetup {
    /// The underlying wgpu instance.
    pub instance: Arc<wgpu::Instance>,
    /// The chose 'adapter'. This corresponds to a physical device.
    pub adapter: Arc<wgpu::Adapter>,
    /// The wpgu device Burn will use. Nb: There can only be one device per adapter.
    pub device: Arc<wgpu::Device>,
    /// The queue Burn commands will be submittd to.
    pub queue: Arc<wgpu::Queue>,
}

/// Create a `WgpuDevice` on an existing `WgpuSetup`. Useful when you want to share
/// a device between CubeCL and other wgpu libraries.
pub fn init_device_on_setup(setup: WgpuSetup, options: RuntimeOptions) -> WgpuDevice {
    let device_id = WgpuDevice::Existing(setup.device.as_ref().global_id());
    let client = create_client_on_setup(setup, options);
    RUNTIME.register(&device_id, client);
    device_id
}

/// Like [`create_setup`], but synchronous.
/// On wasm, it is necessary to use [`init_async`] instead.
pub fn init_device_async<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuSetup {
    #[cfg(target_family = "wasm")]
    panic!("Creating a wgpu setup synchronously is unsupported on wasm. Use init_async instead");

    future::block_on(init_device::<G>(device, options))
}

/// Initialize a client on the given device with the given options. This function is useful to configure the runtime options
/// or to pick a different graphics API.
pub async fn init_device<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuSetup {
    let setup = create_setup_for_device::<G, WgslCompiler>(device).await;
    let return_setup = setup.clone();
    let client = create_client_on_setup(setup, options);
    RUNTIME.register(device, client);
    return_setup
}

fn create_client_on_setup<C: WgpuCompiler>(
    setup: WgpuSetup,
    options: RuntimeOptions,
) -> ComputeClient<WgpuServer<C>, MutexComputeChannel<WgpuServer<C>>> {
    let limits = setup.device.limits();
    let mem_props = MemoryDeviceProperties {
        max_page_size: limits.max_storage_buffer_binding_size as u64,
        alignment: WgpuStorage::ALIGNMENT.max(limits.min_storage_buffer_offset_alignment as u64),
    };

    let memory_management = {
        let device = setup.device.clone();
        let mem_props = mem_props.clone();
        let config = options.memory_config;
        let storage = WgpuStorage::new(device.clone());
        MemoryManagement::from_configuration(storage, mem_props, config)
    };
    let server = WgpuServer::new(
        memory_management,
        setup.device.clone(),
        setup.queue,
        options.tasks_max,
    );
    let channel = MutexComputeChannel::new(server);

    let features = setup.adapter.features();
    let mut device_props = DeviceProperties::new(&[], mem_props);
    if features.contains(wgpu::Features::SUBGROUP) {
        device_props.register_feature(Feature::Subcube);
    }
    C::register_features(&setup.adapter, &setup.device, &mut device_props);
    ComputeClient::new(channel, device_props)
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice).
async fn create_setup_for_device<G: GraphicsApi, C: WgpuCompiler>(
    device: &WgpuDevice,
) -> WgpuSetup {
    let (instance, adapter) = request_adapter::<G>(device).await;
    let (device, queue) = C::request_device(&adapter).await;

    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        adapter.get_info()
    );

    WgpuSetup {
        instance: Arc::new(instance),
        adapter: Arc::new(adapter),
        device: Arc::new(device),
        queue: Arc::new(queue),
    }
}

#[cfg(target_family = "wasm")]
async fn select_adapter<G: GraphicsApi>(_device: &WgpuDevice) -> (wgpu::Instance, wgpu::Adapter) {
    let instance = wgpu::Instance::default();

    instance
        .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
        .await
        .unwrap()
}

#[cfg(not(target_family = "wasm"))]
async fn request_adapter<G: GraphicsApi>(device: &WgpuDevice) -> (wgpu::Instance, wgpu::Adapter) {
    use wgpu::{DeviceType, RequestAdapterOptions};

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: G::backend().into(),
        ..Default::default()
    });
    let mut adapters_other = Vec::new();
    let mut adapters = Vec::new();

    instance
        .enumerate_adapters(G::backend().into())
        .into_iter()
        .for_each(|adapter| {
            let device_type = adapter.get_info().device_type;

            if let DeviceType::Other = device_type {
                adapters_other.push(adapter);
                return;
            }

            let is_same_type = match device {
                WgpuDevice::DiscreteGpu(_) => device_type == DeviceType::DiscreteGpu,
                WgpuDevice::IntegratedGpu(_) => device_type == DeviceType::IntegratedGpu,
                WgpuDevice::VirtualGpu(_) => device_type == DeviceType::VirtualGpu,
                WgpuDevice::Cpu => device_type == DeviceType::Cpu,
                #[allow(deprecated)]
                WgpuDevice::DefaultDevice | WgpuDevice::BestAvailable => true,
                WgpuDevice::Existing(_) => {
                    unreachable!("Cannot select an adapter for an existing device.")
                }
            };

            if is_same_type {
                adapters.push(adapter);
            }
        });

    fn select(
        num: usize,
        error: &str,
        mut adapters: Vec<wgpu::Adapter>,
        mut adapters_other: Vec<wgpu::Adapter>,
    ) -> wgpu::Adapter {
        if adapters.len() <= num {
            if adapters_other.len() <= num {
                panic!(
                    "{}, adapters {:?}, other adapters {:?}",
                    error,
                    adapters
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                    adapters_other
                        .into_iter()
                        .map(|adapter| adapter.get_info())
                        .collect::<Vec<_>>(),
                );
            }

            return adapters_other.remove(num);
        }

        adapters.remove(num)
    }
    #[allow(deprecated)]
    let override_device = if matches!(
        device,
        WgpuDevice::DefaultDevice | WgpuDevice::BestAvailable
    ) {
        get_device_override()
    } else {
        None
    };

    let device = override_device.unwrap_or_else(|| device.clone());

    let adapter = match device {
        WgpuDevice::DiscreteGpu(num) => select(
            num,
            "No Discrete GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::IntegratedGpu(num) => select(
            num,
            "No Integrated GPU device found",
            adapters,
            adapters_other,
        ),
        WgpuDevice::VirtualGpu(num) => {
            select(num, "No Virtual GPU device found", adapters, adapters_other)
        }
        WgpuDevice::Cpu => select(0, "No CPU device found", adapters, adapters_other),
        #[allow(deprecated)]
        WgpuDevice::DefaultDevice | WgpuDevice::BestAvailable => instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("No possible adapter available for backend. Falling back to first available."),
        WgpuDevice::Existing(_) => unreachable!("Cannot select an adapter for an existing device."),
    };

    log::info!("Using adapter {:?}", adapter.get_info());

    (instance, adapter)
}

fn get_device_override() -> Option<WgpuDevice> {
    // If BestAvailable, check if we should instead construct as
    // if a specific device was specified.
    std::env::var("CUBECL_WGPU_DEFAULT_DEVICE")
        .ok()
        .and_then(|var| {
            let override_device = if let Some(inner) = var.strip_prefix("DiscreteGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDevice::DiscreteGpu)
            } else if let Some(inner) = var.strip_prefix("IntegratedGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDevice::IntegratedGpu)
            } else if let Some(inner) = var.strip_prefix("VirtualGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDevice::VirtualGpu)
            } else if var == "Cpu" {
                Some(WgpuDevice::Cpu)
            } else {
                None
            };

            if override_device.is_none() {
                log::warn!("Unknown CUBECL_WGPU_DEVICE override {var}");
            }
            override_device
        })
}
