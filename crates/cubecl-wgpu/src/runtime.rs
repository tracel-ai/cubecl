use crate::{
    AutoCompiler, AutoGraphicsApi, GraphicsApi, WgpuDevice, backend, compute::WgpuServer,
    contiguous_strides,
};
use cubecl_common::device::{Device, DeviceContext, DeviceState};
use cubecl_common::stream_id::StreamId;
use cubecl_common::{future, profile::TimingMethod};
use cubecl_core::server::Handle;
use cubecl_core::{Runtime, ir::TargetProperties};
use cubecl_core::{ir::LineSize, server::ServerUtilities};
use cubecl_ir::{DeviceProperties, HardwareProperties, MemoryDeviceProperties};
pub use cubecl_runtime::memory_management::MemoryConfiguration;
use cubecl_runtime::{
    client::ComputeClient,
    logging::{ProfileLevel, ServerLogger},
};
use wgpu::{InstanceFlags, RequestAdapterOptions};

/// Runtime that uses the [wgpu] crate with the wgsl compiler. This is used in the Wgpu backend.
/// For advanced configuration, use [`init_setup`] to pass in runtime options or to select a
/// specific graphics API.
#[derive(Debug)]
pub struct WgpuRuntime;

impl DeviceState for WgpuServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let device = WgpuDevice::from_id(device_id);
        let setup = future::block_on(create_setup_for_device(&device, AutoGraphicsApi::backend()));
        create_server(setup, RuntimeOptions::default())
    }
}

impl Runtime for WgpuRuntime {
    type Compiler = AutoCompiler;
    type Server = WgpuServer;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(client: &ComputeClient<Self>) -> &'static str {
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

    fn supported_line_sizes() -> &'static [LineSize] {
        #[cfg(feature = "msl")]
        {
            &[8, 4, 2, 1]
        }
        #[cfg(not(feature = "msl"))]
        {
            &[4, 2, 1]
        }
    }

    fn max_global_line_size() -> LineSize {
        4
    }

    fn max_cube_count() -> (u32, u32, u32) {
        let max_dim = u16::MAX as u32;
        (max_dim, max_dim, max_dim)
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

    fn target_properties() -> TargetProperties {
        TargetProperties {
            // Values are irrelevant, since no wgsl backends currently support manual mma
            mma: Default::default(),
        }
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
/// Useful when you want to share a device between `CubeCL` and other wgpu-dependent libraries.
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
    let server = create_server(setup, options);
    let _ = ComputeClient::<WgpuRuntime>::init(&device_id, server);
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
    let server = create_server(setup, options);
    let _ = ComputeClient::<WgpuRuntime>::init(device, server);
    return_setup
}

/// Register an external wgpu resource.
///
/// Ownership of the resource is transferred to CubeCL. The resource will be dropped
/// when all references to the returned handle are released and memory cleanup runs.
///
/// The caller must ensure:
/// - The buffer has compatible usage flags (`STORAGE | COPY_SRC | COPY_DST`)
/// - Any pending GPU operations on the buffer are complete before registration
///
/// # Example
/// ```no_run
/// # use cubecl_wgpu::{WgpuResource, WgpuDevice, register_external};
/// # use cubecl_common::stream_id::StreamId;
/// # fn example(device: &WgpuDevice, my_buffer: wgpu::Buffer, stream_id: StreamId) {
/// let resource = WgpuResource::new(my_buffer, 0, 64);
/// let handle = register_external(device, resource, stream_id);
/// # }
/// ```
pub fn register_external(
    device: &WgpuDevice,
    resource: crate::WgpuResource,
    stream_id: StreamId,
) -> Handle {
    let context = DeviceContext::<WgpuServer>::locate(device);
    let mut server = context.lock();
    cubecl_runtime::server::ComputeServer::register_external(&mut *server, resource, stream_id)
}

/// Immediately unregister an external resource.
///
/// The caller must ensure all GPU operations using this resource have completed before this call.
/// The handle is consumed and becomes invalid. Any handle clones should not be used after this call.
///
/// Returns the resource if found, allowing the caller to use or drop it.
///
/// # Example
/// ```no_run
/// # use cubecl_wgpu::{WgpuDevice, unregister_external};
/// # use cubecl_core::server::Handle;
/// # use cubecl_common::stream_id::StreamId;
/// # fn example(device: &WgpuDevice, handle: Handle, stream_id: StreamId) {
/// if let Some(resource) = unregister_external(device, handle, stream_id) {
///     let buffer = resource.buffer; // Get the wgpu::Buffer back
/// }
/// # }
/// ```
pub fn unregister_external(
    device: &WgpuDevice,
    handle: Handle,
    stream_id: StreamId,
) -> Option<crate::WgpuResource> {
    let context = DeviceContext::<WgpuServer>::locate(device);
    let mut server = context.lock();
    cubecl_runtime::server::ComputeServer::unregister_external(&mut *server, &handle, stream_id)
}

pub(crate) fn create_server(setup: WgpuSetup, options: RuntimeOptions) -> WgpuServer {
    let limits = setup.device.limits();
    let mut adapter_limits = setup.adapter.limits();

    // Workaround: WebGPU reports some "fake" subgroup info atm, as it's not really supported yet.
    // However, some algorithms do rely on having this information eg. cubecl-reduce uses max subgroup size _even_ when
    // subgroups aren't used. For now, just override with the maximum range of subgroups possible.
    if adapter_limits.min_subgroup_size == 0 && adapter_limits.max_subgroup_size == 0 {
        // There is in theory nothing limiting the size to go below 8 but in practice 8 is the minimum found anywhere.
        adapter_limits.min_subgroup_size = 8;
        // This is a hard limit of GPU APIs (subgroup ballot returns 4 * 32 bits).
        adapter_limits.max_subgroup_size = 128;
    }

    let mem_props = MemoryDeviceProperties {
        max_page_size: limits.max_storage_buffer_binding_size as u64,
        alignment: limits.min_storage_buffer_offset_alignment as u64,
    };
    let max_count = adapter_limits.max_compute_workgroups_per_dimension;
    let hardware_props = HardwareProperties {
        load_width: 128,
        // On Apple Silicon, the plane size is 32,
        // though the minimum and maximum differ.
        // https://github.com/gpuweb/gpuweb/issues/3950
        #[cfg(apple_silicon)]
        plane_size_min: 32,
        #[cfg(not(apple_silicon))]
        plane_size_min: adapter_limits.min_subgroup_size,
        #[cfg(apple_silicon)]
        plane_size_max: 32,
        #[cfg(not(apple_silicon))]
        plane_size_max: adapter_limits.max_subgroup_size,
        // wgpu uses an additional buffer for variable-length buffers,
        // so we have to use one buffer less on our side to make room for that wgpu internal buffer.
        // See: https://github.com/gfx-rs/wgpu/blob/a9638c8e3ac09ce4f27ac171f8175671e30365fd/wgpu-hal/src/metal/device.rs#L799
        max_bindings: limits
            .max_storage_buffers_per_shader_stage
            .saturating_sub(1),
        max_shared_memory_size: limits.max_compute_workgroup_storage_size as usize,
        max_cube_count: (max_count, max_count, max_count),
        max_units_per_cube: adapter_limits.max_compute_invocations_per_workgroup,
        max_cube_dim: (
            adapter_limits.max_compute_workgroup_size_x,
            adapter_limits.max_compute_workgroup_size_y,
            adapter_limits.max_compute_workgroup_size_z,
        ),
        num_streaming_multiprocessors: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        num_cpu_cores: None, // TODO: Check if device is CPU.
    };

    let mut compilation_options = Default::default();

    let features = setup.adapter.features();

    let time_measurement = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        TimingMethod::Device
    } else {
        TimingMethod::System
    };

    let mut device_props = DeviceProperties::new(
        Default::default(),
        mem_props.clone(),
        hardware_props,
        time_measurement,
    );

    #[cfg(not(all(target_os = "macos", feature = "msl")))]
    {
        if features.contains(wgpu::Features::SUBGROUP)
            && setup.adapter.get_info().device_type != wgpu::DeviceType::Cpu
        {
            use cubecl_ir::features::Plane;

            device_props.features.plane.insert(Plane::Ops);
        }
    }

    #[cfg(any(feature = "spirv", feature = "msl"))]
    device_props
        .features
        .plane
        .insert(cubecl_ir::features::Plane::NonUniformControlFlow);

    backend::register_features(&setup.adapter, &mut device_props, &mut compilation_options);

    let logger = alloc::sync::Arc::new(ServerLogger::default());

    WgpuServer::new(
        mem_props,
        options.memory_config,
        compilation_options,
        setup.device.clone(),
        setup.queue,
        options.tasks_max,
        setup.backend,
        time_measurement,
        ServerUtilities::new(device_props, logger, setup.backend),
    )
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
