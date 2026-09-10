use std::marker::PhantomData;

use crate::WgpuCompiler;
use crate::{
    AutoCompiler, AutoGraphicsApi, GraphicsApi, WgpuBackend, WgpuDevice, WgpuDeviceKind, backend,
    compute::WgpuServer, contiguous_strides,
};
use cubecl_common::device::{Device, DeviceService, ServiceId};
use cubecl_common::profile::TimingMethod;
use cubecl_core::device::{DeviceId, ServerUtilitiesHandle};
use cubecl_core::ir::TargetProperties;
use cubecl_core::server::ServerUtilities;
use cubecl_core::zspace::{Shape, Strides};
use cubecl_environment::future;
use cubecl_ir::{DeviceIdentity, DeviceProperties, HardwareProperties, MemoryDeviceProperties};
use cubecl_server::allocator::ContiguousMemoryLayoutPolicy;
#[cfg(not(feature = "vulkan-validate"))]
use cubecl_server::logging::ProfileLevel;
pub use cubecl_server::memory_management::MemoryConfiguration;
use cubecl_server::runtime::Runtime;
use cubecl_server::{client::Client, logging::ServerLogger};
use wgpu::{InstanceFlags, RequestAdapterOptions};

/// Runtime that uses the [wgpu] crate with the wgsl compiler. This is used in the Wgpu backend.
/// For advanced configuration, use [`init_setup`] to pass in runtime options or to select a
/// specific graphics API.
#[derive(Debug)]
pub struct WgpuRuntime<Compiler = AutoCompiler> {
    _p: PhantomData<Compiler>,
}

impl<C> Clone for WgpuRuntime<C> {
    fn clone(&self) -> Self {
        Self { _p: self._p }
    }
}

impl<C: WgpuCompiler> DeviceService for WgpuServer<C> {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let device = WgpuDevice::from_id(device_id);
        let setup = future::block_on(create_setup_for_device(
            &device,
            resolve_backend(device.backend),
        ));
        create_server(setup, RuntimeOptions::default(), device_id)
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities.clone() as ServerUtilitiesHandle
    }
}

impl<C: WgpuCompiler> Runtime for WgpuRuntime<C> {
    type Server = WgpuServer<C>;
    type Device = WgpuDevice;

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }

        for (&expected, &stride) in contiguous_strides(shape).iter().zip(strides.iter()) {
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

    fn enumerate_devices(type_id: u16) -> Vec<DeviceId> {
        // The devices of that type on `Auto`, whose id is index zero with no
        // graphics API in the top bits.
        Self::enumerate_devices_like(DeviceId::new(type_id, 0))
    }

    fn is_available() -> bool {
        // A software rasterizer — lavapipe, llvmpipe, WARP — enumerates as
        // `WgpuDeviceKind::Cpu`. It runs, but a machine with nothing else is
        // better served by a native CPU runtime, so wgpu does not claim it;
        // a caller who wants it still names it.
        let gpu = [
            WgpuDeviceKind::DiscreteGpu(0),
            WgpuDeviceKind::IntegratedGpu(0),
            WgpuDeviceKind::VirtualGpu(0),
            WgpuDeviceKind::Other(0),
        ]
        .map(|kind| WgpuDevice::new(kind).to_id().type_id);

        Self::enumerate_all_devices()
            .iter()
            .any(|device| gpu.contains(&device.type_id))
    }

    /// Each adapter once, on whichever graphics API [`WgpuBackend::Auto`]
    /// settles on — which is what a client created lazily resolves these ids
    /// against. The same adapter pinned to another API is a device a caller
    /// names, not one more to count.
    fn enumerate_all_devices() -> Vec<DeviceId> {
        adapters_on(WgpuBackend::Auto)
    }

    fn enumerate_devices_like(device_id: DeviceId) -> Vec<DeviceId> {
        let device = WgpuDevice::from_id(device_id);
        let reachable = adapters_on(device.backend);

        match device.kind {
            // One per graphics API, standing for whichever adapter it lands
            // on: its own only peer, there as soon as the API has anything.
            // Listing the adapters beside it would hand one of them a second
            // client under another id.
            WgpuDeviceKind::DefaultDevice if !reachable.is_empty() => alloc::vec![device_id],
            _ => reachable
                .into_iter()
                .filter(|id| id.type_id == device_id.type_id)
                .collect(),
        }
    }

    /// The browser hands out one adapter without saying what it is, so a kind
    /// there is only a power preference — and `request_adapter` honors the
    /// low-power one as well as the rest.
    #[cfg(target_family = "wasm")]
    fn find_device(device_id: DeviceId) -> Result<(), usize> {
        let device = WgpuDevice::from_id(device_id);
        let peers = Self::enumerate_devices_like(device_id);

        let low_power = device.kind == WgpuDeviceKind::IntegratedGpu(0)
            && !adapters_on(device.backend).is_empty();

        match peers.contains(&device_id) || low_power {
            true => Ok(()),
            false => Err(peers.len()),
        }
    }
}

/// The ids of the adapters `backend` reaches, pinned the way it is.
///
/// Those of the one graphics API it settles on, so each id resolves in
/// `WgpuServer::init` to the adapter it was listed for. A device brought up on
/// another API through [`init_setup`] is reached through the client that call
/// hands back, not here.
fn adapters_on(backend: WgpuBackend) -> Vec<DeviceId> {
    // WebGPU only supports a single device currently, and only the browser's
    // own API reaches it.
    #[cfg(target_family = "wasm")]
    let ids = match backend {
        WgpuBackend::Auto | WgpuBackend::WebGpu => vec![DeviceId::new(0, 0)],
        _ => Vec::new(),
    };

    #[cfg(not(target_family = "wasm"))]
    let ids = settle(backend)
        .map(|(_, adapters)| adapter_device_ids(adapters))
        .unwrap_or_default();

    ids.into_iter()
        .map(|id| WgpuDevice::from_id(id).on(backend).to_id())
        .collect()
}

/// The graphics API `backend` settles on, and the adapters this machine has
/// there: the first of its candidates with a GPU, or failing that, the first
/// with anything at all.
///
/// A software rasterizer is not reason enough to stop. Where Vulkan offers
/// only lavapipe and `OpenGL` the real GPU — a VM passing it through, say —
/// stopping at Vulkan leaves that GPU unreachable through `Auto`, and wgpu
/// declining a machine it could have served.
#[cfg(not(target_family = "wasm"))]
fn settle(backend: WgpuBackend) -> Option<(wgpu::Backend, Vec<wgpu::Adapter>)> {
    let mut software_only = None;

    for api in backend_candidates(backend) {
        let adapters = enumerate_all_adapters(instance_for(api), api);

        if adapters
            .iter()
            .any(|adapter| adapter.get_info().device_type != wgpu::DeviceType::Cpu)
        {
            return Some((api, adapters));
        }

        if software_only.is_none() && !adapters.is_empty() {
            software_only = Some((api, adapters));
        }
    }

    software_only
}

/// The `wgpu` backends to try for a [`WgpuBackend`], best first.
///
/// A pinned one is the only candidate — that is what pinning it means.
fn backend_candidates(backend: WgpuBackend) -> alloc::vec::Vec<wgpu::Backend> {
    match backend {
        WgpuBackend::Auto => AutoGraphicsApi::chain(),
        WgpuBackend::Vulkan => alloc::vec![wgpu::Backend::Vulkan],
        WgpuBackend::Metal => alloc::vec![wgpu::Backend::Metal],
        WgpuBackend::Dx12 => alloc::vec![wgpu::Backend::Dx12],
        WgpuBackend::Gl => alloc::vec![wgpu::Backend::Gl],
        WgpuBackend::WebGpu => alloc::vec![wgpu::Backend::BrowserWebGpu],
    }
}

/// An instance limited to one graphics API, for asking what it has.
#[cfg(not(target_family = "wasm"))]
fn instance_for(backend: wgpu::Backend) -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

/// The graphics API a device on `backend` comes up on.
///
/// The one pinned, where one is. Otherwise the first of the chain
/// this machine has an adapter for — which is what makes `Auto` mean Vulkan
/// wherever Vulkan exists, and the next thing where it does not.
pub(crate) fn resolve_backend(backend: WgpuBackend) -> wgpu::Backend {
    #[cfg(not(target_family = "wasm"))]
    if let Some((api, _)) = settle(backend) {
        return api;
    }

    // Nothing answered: hand back the first anyway, so the failure is the
    // setup's own rather than a silent fallback to some other API.
    backend_candidates(backend)[0]
}

/// The `DeviceId` addressing each adapter, in enumeration order.
///
/// Every device type counts from zero on its own: `WgpuDevice::DiscreteGpu(n)`
/// is the nth *discrete* adapter, not the nth adapter overall, so a single
/// counter over the mixed list hands out ids for devices that do not exist.
/// `Cpu` carries no index in `WgpuDevice`, so it stays at zero.
#[cfg(not(target_family = "wasm"))]
fn adapter_device_ids(adapters: Vec<wgpu::Adapter>) -> Vec<DeviceId> {
    let mut next = [0u16; 7];

    adapters
        .into_iter()
        .map(|adapter| {
            let type_id = match adapter.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::VirtualGpu => 2,
                wgpu::DeviceType::Cpu => 3,
                wgpu::DeviceType::Other => 6,
            };

            // Only the indexed kinds have a counter; the rest are always zero.
            let index = match next.get_mut(type_id as usize).filter(|_| type_id != 3) {
                Some(next) => {
                    let index = *next;
                    *next += 1;
                    index
                }
                None => 0,
            };

            DeviceId::new(type_id, index)
        })
        .collect()
}

#[cfg(not(target_family = "wasm"))]
fn enumerate_all_adapters(instance: wgpu::Instance, backend: wgpu::Backend) -> Vec<wgpu::Adapter> {
    // `enumerate_adapters` is now async & available on WebGPU
    cubecl_environment::future::block_on(instance.enumerate_adapters(backend.into()))
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
        const DEFAULT_MAX_TASKS: usize = 32;
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

    let device_id = WgpuDevice::new(WgpuDeviceKind::Existing(device_id));
    let server = create_server::<AutoCompiler>(setup, options, device_id.to_id());
    let _ = Client::init(device_id.to_id(), server);
    device_id
}

/// Like [`init_setup_async`], but synchronous.
/// On wasm, it is necessary to use [`init_setup_async`] instead.
///
/// A device brought up on a `G` other than [`AutoGraphicsApi`] is reached
/// through the client this initializes, and is not among the devices
/// [`Runtime::enumerate_devices`] lists: those ids index the auto backend's
/// adapters, which is what a client created lazily resolves them against.
///
/// # Panics
///
/// Where `device` pins a graphics API and `G` names another: see
/// [`init_setup_async`].
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
///
/// A device pinned to a graphics API comes up on that API: through
/// [`AutoGraphicsApi`], or through the `G` naming the same one.
///
/// # Panics
///
/// Where `device` pins a graphics API and `G` names another. The client is
/// registered under the device's id, and a pinned id promises its API to
/// every caller who reaches for that client afterwards.
pub async fn init_setup_async<G: GraphicsApi>(
    device: &WgpuDevice,
    options: RuntimeOptions,
) -> WgpuSetup {
    let backend = G::backend_for(device);

    if device.backend != WgpuBackend::Auto {
        let pinned = resolve_backend(device.backend);
        assert_eq!(
            backend, pinned,
            "{device:?} is pinned to {pinned:?}, and cannot be set up on {backend:?}"
        );
    }

    let setup = create_setup_for_device(device, backend).await;
    let return_setup = setup.clone();
    let server = create_server::<AutoCompiler>(setup, options, device.to_id());
    let _ = Client::init(device.to_id(), server);
    return_setup
}

/// The runtime name for `backend`, naming the compiler that serves it.
fn runtime_name(backend: wgpu::Backend) -> &'static str {
    match backend {
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

pub(crate) fn create_server<C: WgpuCompiler>(
    setup: WgpuSetup,
    options: RuntimeOptions,
    device_id: DeviceId,
) -> WgpuServer<C> {
    let limits = setup.device.limits();
    let adapter_limits = setup.adapter.limits();
    let mut adapter_info = setup.adapter.get_info();

    // Workaround: WebGPU reports some "fake" subgroup info atm, as it's not really supported yet.
    // However, some algorithms do rely on having this information eg. cubecl-reduce uses max subgroup size _even_ when
    // subgroups aren't used. For now, just override with the maximum range of subgroups possible.
    if adapter_info.subgroup_min_size == 0 && adapter_info.subgroup_max_size == 0 {
        // There is in theory nothing limiting the size to go below 8 but in practice 8 is the minimum found anywhere.
        adapter_info.subgroup_min_size = 8;
        // This is a hard limit of GPU APIs (subgroup ballot returns 4 * 32 bits).
        adapter_info.subgroup_max_size = 128;
    }

    let mem_props = MemoryDeviceProperties {
        max_page_size: limits.max_storage_buffer_binding_size,
        alignment: limits.min_uniform_buffer_offset_alignment as u64,
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
        plane_size_min: adapter_info.subgroup_min_size,
        #[cfg(apple_silicon)]
        plane_size_max: 32,
        #[cfg(not(apple_silicon))]
        plane_size_max: adapter_info.subgroup_max_size,
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
        last_level_cache_size: None,
        max_vector_size: 4,
        // Init later if extension is enabled
        cube_mma_reserved_shared_memory: 0,
    };

    let mut compilation_options = Default::default();

    let features = setup.adapter.features();

    let time_measurement = if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        TimingMethod::Device
    } else {
        TimingMethod::System
    };

    // The adapter's vendor/device pair, which is what `WgpuServer` keys its
    // SPIR-V store on. Reported unconditionally, even in a WGSL-only build that
    // persists no compiled code: measurement caches (autotune, throughput) are
    // namespaced by neither vendor nor device, so this string is the only thing
    // that can tell one adapter's measurements from another's.
    let fingerprint = format!("spirv_{}_{}", adapter_info.vendor, adapter_info.device);

    let mut device_props = DeviceProperties::new(
        Default::default(),
        mem_props,
        hardware_props,
        time_measurement,
        DeviceIdentity {
            name: adapter_info.name.clone(),
            fingerprint,
        },
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

    backend::register_features(
        &setup.adapter,
        &mut device_props,
        &mut compilation_options,
        &options.memory_config,
    );

    let logger = alloc::sync::Arc::new(ServerLogger::default());

    let allocator = ContiguousMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
    WgpuServer::new(
        device_props.memory.clone(),
        options.memory_config,
        compilation_options,
        setup.device.clone(),
        setup.queue,
        options.tasks_max,
        setup.backend,
        time_measurement,
        ServerUtilities::new(
            ServiceId::of::<WgpuServer<C>>(device_id),
            runtime_name(setup.backend),
            device_props,
            WgpuRuntime::<C>::target_properties(),
            logger,
            allocator,
        ),
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
        "Created wgpu compute server on device {:?}",
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
    #[cfg(not(feature = "vulkan-validate"))]
    let instance_flags = {
        let debug = ServerLogger::default();
        // Debug/validation layers cost real per-dispatch CPU time, so only
        // source-level compilation logging (`full`) opts into them — `basic`
        // is passive name-only logging and must not change how kernels run.
        match (debug.profile_level(), debug.compilation_source_activated()) {
            (Some(ProfileLevel::Full), _) => InstanceFlags::advanced_debugging(),
            (_, true) => InstanceFlags::debugging(),
            (_, false) => InstanceFlags::default(),
        }
    };
    #[cfg(feature = "vulkan-validate")]
    let instance_flags = InstanceFlags::advanced_debugging();
    log::debug!("{instance_flags:?}");
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend.into(),
        flags: instance_flags,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    // The variable names a device, not a graphics API, so a caller who pinned
    // one keeps it.
    let override_device = match device.kind {
        WgpuDeviceKind::DefaultDevice => get_device_override().map(|kind| WgpuDevice {
            kind,
            backend: device.backend,
        }),
        _ => None,
    };

    let device = override_device.unwrap_or_else(|| device.clone());

    let adapter = match device.kind {
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::DiscreteGpu(num) => {
            select_from_adapter_list(
                num,
                "No Discrete GPU device found",
                &instance,
                &device,
                backend,
            )
            .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::IntegratedGpu(num) => {
            select_from_adapter_list(
                num,
                "No Integrated GPU device found",
                &instance,
                &device,
                backend,
            )
            .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::VirtualGpu(num) => {
            select_from_adapter_list(
                num,
                "No Virtual GPU device found",
                &instance,
                &device,
                backend,
            )
            .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::Other(num) => {
            select_from_adapter_list(num, "No Other device found", &instance, &device, backend)
                .await
        }
        #[cfg(not(target_family = "wasm"))]
        WgpuDeviceKind::Cpu => {
            select_from_adapter_list(0, "No CPU device found", &instance, &device, backend).await
        }
        #[cfg(target_family = "wasm")]
        WgpuDeviceKind::IntegratedGpu(_) => {
            request_adapter_with_preference(&instance, wgpu::PowerPreference::LowPower).await
        }
        WgpuDeviceKind::Existing(_) => {
            unreachable!("Cannot select an adapter for an existing device.")
        }
        _ => {
            request_adapter_with_preference(&instance, wgpu::PowerPreference::HighPerformance).await
        }
    };

    (instance, adapter)
}

async fn request_adapter_with_preference(
    instance: &wgpu::Instance,
    power_preference: wgpu::PowerPreference,
) -> wgpu::Adapter {
    instance
        .request_adapter(&RequestAdapterOptions {
            power_preference,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..RequestAdapterOptions::default()
        })
        .await
        .expect("No possible adapter available for backend. Falling back to first available.")
}

#[cfg(not(target_family = "wasm"))]
async fn select_from_adapter_list(
    num: usize,
    error: &str,
    instance: &wgpu::Instance,
    device: &WgpuDevice,
    backend: wgpu::Backend,
) -> wgpu::Adapter {
    // A kind is what the graphics API reports, and nothing stands in for it:
    // `OpenGL` calling a GPU `Other` makes it `Other(n)` there, not a discrete
    // GPU by another name. Anything looser selects adapters `find_device` says
    // the machine does not have, and two ids end up on one adapter.
    let adapters = instance.enumerate_adapters(backend.into()).await;
    let found = adapters
        .iter()
        .map(|adapter| adapter.get_info())
        .collect::<Vec<_>>();

    let is_same_type = |adapter: &wgpu::Adapter| {
        let device_type = adapter.get_info().device_type;

        match device.kind {
            WgpuDeviceKind::DiscreteGpu(_) => device_type == wgpu::DeviceType::DiscreteGpu,
            WgpuDeviceKind::IntegratedGpu(_) => device_type == wgpu::DeviceType::IntegratedGpu,
            WgpuDeviceKind::VirtualGpu(_) => device_type == wgpu::DeviceType::VirtualGpu,
            WgpuDeviceKind::Cpu => device_type == wgpu::DeviceType::Cpu,
            WgpuDeviceKind::Other(_) => device_type == wgpu::DeviceType::Other,
            WgpuDeviceKind::DefaultDevice => true,
            WgpuDeviceKind::Existing(_) => {
                unreachable!("Cannot select an adapter for an existing device.")
            }
        }
    };

    adapters
        .into_iter()
        .filter(is_same_type)
        .nth(num)
        .unwrap_or_else(|| panic!("{error}, adapters {found:?}"))
}

fn get_device_override() -> Option<WgpuDeviceKind> {
    // If BestAvailable, check if we should instead construct as
    // if a specific device was specified.
    std::env::var("CUBECL_WGPU_DEFAULT_DEVICE")
        .ok()
        .and_then(|var| {
            let override_device = if let Some(inner) = var.strip_prefix("DiscreteGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDeviceKind::DiscreteGpu)
            } else if let Some(inner) = var.strip_prefix("IntegratedGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDeviceKind::IntegratedGpu)
            } else if let Some(inner) = var.strip_prefix("VirtualGpu(") {
                inner
                    .strip_suffix(")")
                    .and_then(|s| s.parse().ok())
                    .map(WgpuDeviceKind::VirtualGpu)
            } else if var == "Cpu" {
                Some(WgpuDeviceKind::Cpu)
            } else {
                None
            };

            if override_device.is_none() {
                log::warn!("Unknown CUBECL_WGPU_DEVICE override {var}");
            }
            override_device
        })
}

#[cfg(all(test, not(target_family = "wasm")))]
mod device_tests {
    use super::*;

    const PINNED: [WgpuBackend; 4] = [
        WgpuBackend::Vulkan,
        WgpuBackend::Metal,
        WgpuBackend::Dx12,
        WgpuBackend::Gl,
    ];

    /// One adapter is one device. Listing it again for every graphics API
    /// that reaches it makes a single GPU look like several, and whatever
    /// counts devices — a collective, a transfer between two of them — runs
    /// on hardware that is not there.
    #[test]
    fn each_adapter_is_listed_once() {
        let ids = <WgpuRuntime>::enumerate_all_devices();

        let adapters = settle(WgpuBackend::Auto).map_or(0, |(_, adapters)| adapters.len());

        assert_eq!(ids.len(), adapters);
        for id in ids {
            assert_eq!(WgpuDevice::from_id(id).backend, WgpuBackend::Auto);
        }
    }

    /// A device pinned to an API is found where that API has it, whatever
    /// the API `Auto` settles on has — the kinds differ from one API to the
    /// next, `OpenGL` calling a GPU what Vulkan calls discrete.
    #[test]
    fn a_device_is_found_on_the_api_it_names() {
        for backend in PINNED {
            let reachable = adapters_on(backend);

            for id in reachable.iter() {
                assert_eq!(
                    <WgpuRuntime>::find_device(*id),
                    Ok(()),
                    "{id} on {backend:?}"
                );
            }

            let default = WgpuDevice::new(WgpuDeviceKind::DefaultDevice).on(backend);
            assert_eq!(
                <WgpuRuntime>::find_device(default.to_id()).is_ok(),
                !reachable.is_empty(),
                "the default device on {backend:?}"
            );
        }
    }

    /// The default device stands for whichever adapter it lands on, so it is
    /// its own only peer: the adapters listed beside it would give one of them
    /// a second client, and leaving it out drops the caller from its own list.
    #[test]
    fn the_default_device_is_its_own_only_peer() {
        for backend in PINNED.into_iter().chain([WgpuBackend::Auto]) {
            let default = WgpuDevice::new(WgpuDeviceKind::DefaultDevice)
                .on(backend)
                .to_id();

            let expected = match adapters_on(backend).is_empty() {
                true => Vec::new(),
                false => alloc::vec![default],
            };

            assert_eq!(<WgpuRuntime>::enumerate_devices_like(default), expected);
        }
    }

    /// Setting a pinned device up through `AutoGraphicsApi` keeps its pin —
    /// the natural call, `init_setup` being the only way to pass options.
    #[test]
    fn auto_defers_to_the_api_a_device_pins() {
        for (backend, api) in [
            (WgpuBackend::Vulkan, wgpu::Backend::Vulkan),
            (WgpuBackend::Metal, wgpu::Backend::Metal),
            (WgpuBackend::Dx12, wgpu::Backend::Dx12),
            (WgpuBackend::Gl, wgpu::Backend::Gl),
        ] {
            let device = WgpuDevice::new(WgpuDeviceKind::DefaultDevice).on(backend);

            assert_eq!(AutoGraphicsApi::backend_for(&device), api);
        }
    }

    /// Naming one API for a device pinned to another is refused before any
    /// adapter is asked for: the client lands under the pinned id, and would
    /// hand every later caller the wrong API.
    #[test]
    #[should_panic(expected = "is pinned to Gl")]
    fn a_setup_on_another_api_than_the_pinned_one_is_refused() {
        let device = WgpuDevice::new(WgpuDeviceKind::DefaultDevice).on(WgpuBackend::Gl);

        init_setup::<crate::Vulkan>(&device, RuntimeOptions::default());
    }

    /// A pinned device's peers are those of its own API. The same adapters on
    /// `Auto` are other devices, with other clients.
    #[test]
    fn a_pinned_device_is_enumerated_with_its_own_api() {
        for backend in PINNED {
            for id in adapters_on(backend) {
                let peers = <WgpuRuntime>::enumerate_devices_like(id);

                assert!(peers.contains(&id), "{id} among {peers:?}");
                for peer in peers {
                    assert_eq!(WgpuDevice::from_id(peer).backend, backend);
                }
            }
        }
    }

    /// And one that API does not have is a miss, reported against what it
    /// has of that kind.
    #[test]
    fn an_index_past_the_end_is_not_found_on_any_api() {
        for backend in PINNED.into_iter().chain([WgpuBackend::Auto]) {
            let device = WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(4242)).on(backend);

            let discrete = adapters_on(backend)
                .into_iter()
                .filter(|id| id.type_id == device.to_id().type_id)
                .count();

            assert_eq!(<WgpuRuntime>::find_device(device.to_id()), Err(discrete));
        }
    }
}
