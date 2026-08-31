use crate::{
    compiler::{HipBackend, HipCompilationOptions, HipCompiler},
    compute::{HipServer, context::HipContext},
    device::AmdDevice,
};
use core::ffi::c_int;
use std::sync::OnceLock;

use cubecl_common::{
    device::{Device, DeviceService},
    profile::TimingMethod,
};
use cubecl_core::{
    MemoryConfiguration, Runtime,
    cmma::MatrixLayout,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        ContiguousElements, DeviceIdentity, DeviceProperties, HardwareProperties,
        MemoryDeviceProperties, MmaProperties, TargetProperties, VectorSize, amd::GfxArch,
        features::Plane,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::{
    hip::{
        self,
        arch::AmdWmma,
        mma::{
            HipCmmaCompiler,
            manual::{contiguous_elements_rdna3, contiguous_elements_rdna4},
        },
    },
    register_supported_types,
    shared::{
        Architecture, CompilationOptions, CppSupportedFeatures, register_mma_features,
        register_scaled_mma_features, register_wmma_features,
    },
};
use cubecl_hip_sys::{hipDeviceScheduleSpin, hipGetDeviceCount, hipSetDeviceFlags};
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy, client::ComputeClient, driver::checked,
    logging::ServerLogger,
};
use std::{ffi::CStr, mem::MaybeUninit, sync::Arc};

static AMD_WMMA: OnceLock<Option<AmdWmma>> = OnceLock::new();

/// The values that control how a HIP Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct HipRuntime;

impl DeviceService for HipServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let device = AmdDevice::from_id(device_id);
        let probe = DeviceProbe::of(device.index as i32);

        // Parsed once, here, and handed to whichever backend compiles: the target feature
        // suffix (`xnack`, `sramecc`) is not part of the name the tables are keyed by.
        let gfx = GfxArch::parse(&probe.arch_name);
        let arch = gfx.family();
        // `Runtime::target_properties` is static, so stash what it needs from the device we're
        // initializing. A process mixing RDNA3 and RDNA4 GPUs would see whichever came up first,
        // which is a limitation the static signature can't express anyway.
        let _ = AMD_WMMA.set(gfx.wmma());
        // The architecture table decides the plane size the compiler emits for, so a driver
        // reporting a different one, or a name the table has never seen, would mean every
        // kernel is generated for the wrong wavefront width.
        assert_eq!(
            Some(probe.warp_size),
            gfx.plane_dim(),
            "the driver reports a wavefront of {} for {:?} (reported as {:?}), but the \
             architecture table generates code for {:?}",
            probe.warp_size,
            gfx.name(),
            probe.arch_name,
            gfx.plane_dim(),
        );

        // SAFETY: Calling HIP FFI to set the active device and configure spin-wait scheduling
        // for the current thread. The device index has been validated above by a successful
        // `hipGetDevicePropertiesR0600` call.
        unsafe {
            let status = cubecl_hip_sys::hipSetDevice(device.index as cubecl_hip_sys::hipDevice_t);
            hipSetDeviceFlags(hipDeviceScheduleSpin);
            checked("hipSetDevice", status)
                .expect("the current thread needs its device set before anything is issued");
        }

        // SAFETY: Calling HIP FFI to query device memory info. The pointers to `free` and
        // `total` are valid stack variables cast to mutable pointers; HIP writes the values
        // through them on success (asserted below).
        let max_memory = unsafe {
            let free: usize = 0;
            let total: usize = 0;
            let status = cubecl_hip_sys::hipMemGetInfo(
                &free as *const _ as *mut usize,
                &total as *const _ as *mut usize,
            );
            checked("hipMemGetInfo", status)
                .expect("the memory pools are sized against the device's capacity");
            total
        };
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_memory as u64 / 4,
            alignment: probe.alignment as u64,
        };

        let supported_wmma_combinations =
            HipCmmaCompiler::RocWmma.supported_cmma_combinations(&arch);
        let supported_mma_combinations = hip::supported_mma_combinations(&arch);
        let supported_scaled_mma_combinations = hip::supported_scaled_mma_combinations(&arch);

        let topology = HardwareProperties {
            load_width: 128,
            plane_size_min: probe.warp_size,
            plane_size_max: probe.warp_size,
            max_bindings: crate::device::AMD_MAX_BINDINGS,
            max_shared_memory_size: probe.max_shared_memory,
            max_cube_count: probe.max_cube_count,
            max_units_per_cube: probe.max_units_per_cube,
            max_cube_dim: probe.max_cube_dim,
            // Consumers that size a grid against the machine need this: without
            // it cubek's matmul selectors fall back to `CubeCountStrategy::FromProblem`
            // and the cube count bears no relation to the device it runs on.
            num_streaming_multiprocessors: probe.num_sms,
            num_tensor_cores: None,
            min_tensor_cores_dim: if supported_wmma_combinations.is_empty() {
                None
            } else {
                Some(16)
            },
            num_cpu_cores: None,
            last_level_cache_size: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };

        // The full `gcnArchName`, target-feature suffix included: HIP RTC gets
        // no `--offload-arch`, so the code object it emits carries this exact
        // string and a loader rejects it on a device that differs by so much as
        // `xnack`. Built once here and handed to both the identity and the
        // compilation cache, so the two can never disagree about what a kernel
        // was built for.
        let fingerprint = format!("hip-kernel_{}", probe.arch_name);

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            topology,
            TimingMethod::System,
            DeviceIdentity {
                name: probe.name.clone(),
                fingerprint: fingerprint.clone(),
            },
        );
        register_supported_types(&mut device_props);

        // TODO look into unsafeAtomicAdd (https://github.com/ROCm/HIP/issues/3573120)
        // device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F16)));
        // device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));

        device_props.features.memory_reinterpret = true;
        device_props.features.alignment = true;
        device_props.features.plane.insert(Plane::Ops);
        device_props
            .features
            .plane
            .insert(Plane::NonUniformControlFlow);

        register_wmma_features(supported_wmma_combinations, &mut device_props);
        register_mma_features(supported_mma_combinations, &mut device_props);
        register_scaled_mma_features(supported_scaled_mma_combinations, &mut device_props);

        let comp_opts = HipCompilationOptions {
            cpp: CompilationOptions {
                warp_size: arch.warp_size() as usize,
                supports_features: CppSupportedFeatures {
                    fast_math: true,
                    ..Default::default()
                },
                amd_wmma: gfx.wmma(),
            },
            arch: Some(gfx),
        };
        let hip_ctx = HipContext::new(
            comp_opts,
            device_props.clone(),
            fingerprint,
            HipBackend::default(),
        );
        let logger = Arc::new(ServerLogger::default());
        let policy = PitchedMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
        let utilities = ServerUtilities::new(device_props, logger, (), policy);
        let options = RuntimeOptions::default();

        HipServer::new(
            hip_ctx,
            mem_properties,
            options.memory_config,
            probe.alignment,
            probe.integrated,
            utilities,
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        cubecl_core::server::ComputeServer::utilities(self) as ServerUtilitiesHandle
    }
}

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer;
    type Device = AmdDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "hip"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        // RDNA3 hands every lane the whole `k` range, duplicated across lanes 0-15 / 16-31. RDNA4
        // splits `k` between the two halves instead, so there's no duplication to account for.
        let rdna4 = AMD_WMMA.get().copied().flatten() == Some(AmdWmma::Rdna4);
        let duplication_ab = if rdna4 { 1 } else { 2 };
        TargetProperties {
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::RowMajor,
                register_layout_b: MatrixLayout::ColMajor,
                register_layout_acc: MatrixLayout::ColMajor,
                register_duplication_a: duplication_ab,
                register_duplication_b: duplication_ab,
                register_duplication_acc: 1,
                contiguous_elements: ContiguousElements::new(if rdna4 {
                    contiguous_elements_rdna4
                } else {
                    contiguous_elements_rdna3
                }),
            },
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as cubecl_core::server::ComputeServer>::Info,
    ) -> Vec<cubecl_core::device::DeviceId> {
        fn device_count() -> usize {
            let mut device_count: c_int = 0;
            let result;
            // SAFETY: Calling HIP FFI to get the number of available devices.
            // `device_count` is a valid mutable pointer to a stack-allocated `c_int`.
            unsafe {
                result = hipGetDeviceCount(&mut device_count);
            }
            // No devices rather than an error: a machine with no HIP runtime
            // installed is answering the question, not failing at it.
            match checked("hipGetDeviceCount", result) {
                Ok(()) => device_count.try_into().unwrap_or(0),
                Err(_) => 0,
            }
        }
        (0..device_count())
            .map(|i| DeviceId::new(0, i as u16))
            .collect()
    }
}

/// What the driver says about one AMD device.
///
/// Every field is copied out of the property struct rather than borrowed from
/// it. That struct is a 1.6 KB C type built for the length of one query, and a
/// `&str` into its `gcnArchName` array would outlive it by the whole of
/// `init` — [`CStr::from_ptr`] hands out whatever lifetime is asked of it, so
/// nothing would say so.
struct DeviceProbe {
    /// The full `gcnArchName`, target-feature suffix included. HIP RTC gets no
    /// `--offload-arch`, so the code object it emits carries this exact string
    /// and a loader rejects it on a device that differs by so much as `xnack`.
    arch_name: String,
    /// The marketing name, lossily decoded: a driver returning something that
    /// is not UTF-8 must not take the runtime down over a display string.
    name: String,
    /// The wavefront width, which the architecture table has to agree with.
    warp_size: u32,
    /// What the driver counts as a multiprocessor. On RDNA that is the work
    /// group processor rather than the compute unit -- gfx1151 reports 20 for
    /// its 40 CUs -- which is the right figure either way, since the WGP is
    /// what a cube is scheduled onto. `None` when the driver reports zero,
    /// which is a driver that does not know rather than a device with none.
    num_sms: Option<u32>,
    max_shared_memory: usize,
    max_cube_count: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
    /// The strictest alignment the device asks for, never below 32.
    alignment: usize,
    /// An APU, sharing its memory and IOMMU with the host. The drop queue
    /// flushes more often on one, to keep the GPU off a 0-to-100% transition.
    integrated: bool,
}

impl DeviceProbe {
    /// Ask the driver to describe the device at `index`.
    ///
    /// # Panics
    ///
    /// If the driver cannot describe its own device. [`DeviceService::init`]
    /// returns `Self`, so there is nowhere to report this to and nothing that
    /// follows would be meaningful.
    fn of(index: i32) -> Self {
        // SAFETY: `hipGetDevicePropertiesR0600` initializes the struct on
        // success, which is asserted before anything reads it, and `index` was
        // validated by the `AmdDevice` constructor.
        let props = unsafe {
            let mut props = MaybeUninit::uninit();
            let status = cubecl_hip_sys::hipGetDevicePropertiesR0600(
                props.as_mut_ptr(),
                index as cubecl_hip_sys::hipDevice_t,
            );
            checked("hipGetDevicePropertiesR0600", status).unwrap_or_else(|err| {
                panic!("the driver could not describe device {index}: {err}")
            });
            props.assume_init()
        };

        // SAFETY: both arrays are null-terminated C strings written by the
        // driver, and both are copied out before `props` goes out of scope.
        let (arch_name, name) = unsafe {
            (
                CStr::from_ptr(props.gcnArchName.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
                CStr::from_ptr(props.name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
            )
        };

        Self {
            arch_name,
            name,
            warp_size: props.warpSize as u32,
            num_sms: (props.multiProcessorCount > 0).then_some(props.multiProcessorCount as u32),
            max_shared_memory: props.sharedMemPerBlock,
            max_cube_count: (
                props.maxGridSize[0] as u32,
                props.maxGridSize[1] as u32,
                props.maxGridSize[2] as u32,
            ),
            max_units_per_cube: props.maxThreadsPerBlock as u32,
            max_cube_dim: (
                props.maxThreadsDim[0] as u32,
                props.maxThreadsDim[1] as u32,
                props.maxThreadsDim[2] as u32,
            ),
            // Both are checked: 32 is the floor either way.
            alignment: 32.max(props.textureAlignment).max(props.surfaceAlignment),
            integrated: props.integrated != 0,
        }
    }
}
