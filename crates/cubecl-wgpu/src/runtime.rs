use crate::{
    AutoCompiler, AutoGraphicsApi, GraphicsApi, WgpuDevice, backend,
    compute::{WgpuServer, WgpuStorage},
    contiguous_strides,
};
use cubecl_common::future;
use cubecl_core::{
    AtomicFeature, CubeDim, Feature, Runtime,
    ir::{Elem, FloatKind},
};
pub use cubecl_runtime::memory_management::MemoryConfiguration;
use cubecl_runtime::{
    ComputeRuntime,
    channel::MutexComputeChannel,
    client::ComputeClient,
    id::DeviceId,
    logging::{ProfileLevel, ServerLogger},
};
use cubecl_runtime::{DeviceProperties, memory_management::HardwareProperties};
use cubecl_runtime::{memory_management::MemoryDeviceProperties, storage::ComputeStorage};
use wgpu::{InstanceFlags, RequestAdapterOptions};

/// Runtime that uses the [wgpu] crate with the wgsl compiler. This is used in the Wgpu backend.
/// For advanced configuration, use [`init_setup`] to pass in runtime options or to select a
/// specific graphics API.
#[derive(Debug)]
pub struct WgpuRuntime;

type Server = WgpuServer;

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

impl Runtime for WgpuRuntime {
    type Compiler = AutoCompiler;
    type Server = WgpuServer;

    type Channel = MutexComputeChannel<WgpuServer>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            let setup =
                future::block_on(create_setup_for_device(device, AutoGraphicsApi::backend()));
            create_client_on_setup(setup, RuntimeOptions::default())
        })
    }

    fn name(client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str {
        match client.info() {
            wgpu::Backend::Vulkan => {
                #[cfg(feature = "spirv")]
                return "wgpu<spirv>";

                #[cfg(not(feature = "spirv"))]
                return "wgpu<wgsl>";
            }
            wgpu::Backend::Metal => {
                #[cfg(feature = "msl")]
                return "wgpu<msl>";

                #[cfg(not(feature = "msl"))]
                return "wgpu<wgsl>";
            }
            _ => "wgpu<wgsl>",
        }
    }

    fn supported_line_sizes() -> &'static [u8] {
        #[cfg(feature = "msl")]
        {
            &[8, 4, 2, 1]
        }
        #[cfg(not(feature = "msl"))]
        {
            &[4, 2, 1]
        }
    }

    fn max_cube_count() -> (u32, u32, u32) {
        let max_dim = u16::MAX as u32;
        (max_dim, max_dim, max_dim)
    }

    fn device_id(device: &Self::Device) -> DeviceId {
        #[allow(deprecated)]
        match device {
            WgpuDevice::DiscreteGpu(index) => DeviceId::new(0, *index as u32),
            WgpuDevice::IntegratedGpu(index) => DeviceId::new(1, *index as u32),
            WgpuDevice::VirtualGpu(index) => DeviceId::new(2, *index as u32),
            WgpuDevice::Cpu => DeviceId::new(3, 0),
            WgpuDevice::BestAvailable | WgpuDevice::DefaultDevice => DeviceId::new(4, 0),
            WgpuDevice::Existing(id) => DeviceId::new(5, *id),
        }
    }

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        if shape.is_empty() {
            return true;
        }

        for (expected, &stride) in contiguous_strides(shape).into_iter().zip(strides) {
            if expected != stride {
                return false;
            }
        }

        true
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
        const DEFAULT_MAX_TASKS: usize = 32;

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

/// A complete setup used to run wgpu.
///
/// These can either be created with [`init_setup`] or [`init_setup_async`].
#[derive(Clone, Debug)]
pub struct WgpuSetup {
    /// The underlying wgpu instance.
    pub instance: wgpu::Instance,
    /// The selected 'adapter'. This corresponds to a physical device.
    pub adapter: wgpu::Adapter,
    /// The wgpu device Burn will use. Nb: There can only be one device per adapter.
    pub device: wgpu::Device,
    /// The queue Burn commands will be submitted to.
    pub queue: wgpu::Queue,
    /// The backend used by the setup.
    pub backend: wgpu::Backend,
}

/// Create a [`WgpuDevice`] on an existing [`WgpuSetup`].
/// Useful when you want to share a device between CubeCL and other wgpu-dependent libraries.
///
/// # Note
///
/// Please **do not** to call on the same [`setup`](WgpuSetup) more than once.
///
/// This function generates a new, globally unique ID for the device every time it is called,
/// even if called on the same device multiple times.
pub fn init_device(setup: WgpuSetup, options: RuntimeOptions) -> WgpuDevice {
    use core::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    let device_id = COUNTER.fetch_add(1, Ordering::Relaxed);
    if device_id == u32::MAX {
        core::panic!("Memory ID overflowed");
    }

    let device_id = WgpuDevice::Existing(device_id);
    let client = create_client_on_setup(setup, options);
    RUNTIME.register(&device_id, client);
    device_id
}

/// Like [`init_setup_async`], but synchronous.
/// On wasm, it is necessary to use [`init_setup_async`] instead.
pub fn init_setup<G: GraphicsApi>(device: &WgpuDevice, options: RuntimeOptions) -> WgpuSetup {
    cfg_if::cfg_if! {
        if #[cfg(target_family = "wasm")] {
            let _ = (device, options);
            panic!("Creating a wgpu setup synchronously is unsupported on wasm. Use init_async instead");
        } else {
            future::block_on(init_setup_async::<G>(device, options))
        }
    }
}

/// Initialize a client on the given device with the given options.
/// This function is useful to configure the runtime options
/// or to pick a different graphics API.
pub async fn init_setup_async<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuSetup {
    let setup = create_setup_for_device(device, G::backend()).await;
    let return_setup = setup.clone();
    let client = create_client_on_setup(setup, options);
    RUNTIME.register(device, client);
    return_setup
}

pub(crate) fn create_client_on_setup(
    setup: WgpuSetup,
    options: RuntimeOptions,
) -> ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>> {
    let limits = setup.device.limits();
    let adapter_limits = setup.adapter.limits();

    let mem_props = MemoryDeviceProperties {
        max_page_size: limits.max_storage_buffer_binding_size as u64,
        alignment: WgpuStorage::ALIGNMENT.max(limits.min_storage_buffer_offset_alignment as u64),
    };
    let max_count = adapter_limits.max_compute_workgroups_per_dimension;
    let hardware_props = HardwareProperties {
        // On Apple Silicon, the plane size is 32,
        // though the minimum and maximum differ.
        // https://github.com/gpuweb/gpuweb/issues/3950
        #[cfg(apple_silicon)]
        plane_size_min: 32,
        #[cfg(not(apple_silicon))]
        plane_size_min: adapter_limits.min_subgroup_size,
        plane_size_max: adapter_limits.max_subgroup_size,
        max_bindings: limits.max_storage_buffers_per_shader_stage,
        max_shared_memory_size: limits.max_compute_workgroup_storage_size as usize,
        max_cube_count: CubeDim::new_3d(max_count, max_count, max_count),
        max_units_per_cube: adapter_limits.max_compute_invocations_per_workgroup,
        max_cube_dim: CubeDim::new_3d(
            adapter_limits.max_compute_workgroup_size_x,
            adapter_limits.max_compute_workgroup_size_y,
            adapter_limits.max_compute_workgroup_size_z,
        ),
        num_streaming_multiprocessors: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
    };

    let mut compilation_options = Default::default();

    let features = setup.adapter.features();

    let mut device_props = DeviceProperties::new(&[], mem_props.clone(), hardware_props);

    // Workaround: WebGPU does support subgroups and correctly reports this, but wgpu
    // doesn't plumb through this info. Instead min/max are just reported as 0, which can cause issues.
    // For now just disable subgroups on WebGPU, until this information is added.
    let fake_plane_info =
        adapter_limits.min_subgroup_size == 0 && adapter_limits.max_subgroup_size == 0;

    if features.contains(wgpu::Features::SUBGROUP)
        && setup.adapter.get_info().device_type != wgpu::DeviceType::Cpu
        && !fake_plane_info
    {
        device_props.register_feature(Feature::Plane);
    }
    backend::register_features(&setup.adapter, &mut device_props, &mut compilation_options);

    let server = WgpuServer::new(
        mem_props,
        options.memory_config,
        compilation_options,
        setup.device.clone(),
        setup.queue,
        options.tasks_max,
        setup.backend,
    );
    let channel = MutexComputeChannel::new(server);

    if features.contains(wgpu::Features::SHADER_FLOAT32_ATOMIC) {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F32)));

        device_props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
        device_props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));
    }

    ComputeClient::new(channel, device_props, setup.backend)
}

/// Select the wgpu device and queue based on the provided [device](WgpuDevice) and
/// [backend](wgpu::Backend).
pub(crate) async fn create_setup_for_device(
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> WgpuSetup {
    let (instance, adapter) = request_adapter(device, backend).await;
    let (device, queue) = backend::request_device(&adapter).await;

    log::info!(
        "Created wgpu compute server on device {:?} => {:?}",
        device,
        adapter.get_info()
    );

    WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend,
    }
}

async fn request_adapter(
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> (wgpu::Instance, wgpu::Adapter) {
    let debug = ServerLogger::default();
    let instance_flags = match (debug.profile_level(), debug.compilation_activated()) {
        (Some(ProfileLevel::Full), _) => InstanceFlags::advanced_debugging(),
        (_, true) => InstanceFlags::debugging(),
        (_, false) => InstanceFlags::default(),
    };
    log::debug!("{instance_flags:?}");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: backend.into(),
        flags: instance_flags,
        ..Default::default()
    });

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
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::DiscreteGpu(num) => select_from_adapter_list(
            num,
            "No Discrete GPU device found",
            &instance,
            &device,
            backend,
        ),
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::IntegratedGpu(num) => select_from_adapter_list(
            num,
            "No Integrated GPU device found",
            &instance,
            &device,
            backend,
        ),
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::VirtualGpu(num) => select_from_adapter_list(
            num,
            "No Virtual GPU device found",
            &instance,
            &device,
            backend,
        ),
        #[cfg(not(target_family = "wasm"))]
        WgpuDevice::Cpu => {
            select_from_adapter_list(0, "No CPU device found", &instance, &device, backend)
        }
        WgpuDevice::Existing(_) => {
            unreachable!("Cannot select an adapter for an existing device.")
        }
        _ => instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("No possible adapter available for backend. Falling back to first available."),
    };

    log::info!("Using adapter {:?}", adapter.get_info());

    (instance, adapter)
}

#[cfg(not(target_family = "wasm"))]
fn select_from_adapter_list(
    num: usize,
    error: &str,
    instance: &wgpu::Instance,
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> wgpu::Adapter {
    let mut adapters_other = Vec::new();
    let mut adapters = Vec::new();

    instance
        .enumerate_adapters(backend.into())
        .into_iter()
        .for_each(|adapter| {
            let device_type = adapter.get_info().device_type;

            if let wgpu::DeviceType::Other = device_type {
                adapters_other.push(adapter);
                return;
            }

            let is_same_type = match device {
                WgpuDevice::DiscreteGpu(_) => device_type == wgpu::DeviceType::DiscreteGpu,
                WgpuDevice::IntegratedGpu(_) => device_type == wgpu::DeviceType::IntegratedGpu,
                WgpuDevice::VirtualGpu(_) => device_type == wgpu::DeviceType::VirtualGpu,
                WgpuDevice::Cpu => device_type == wgpu::DeviceType::Cpu,
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
