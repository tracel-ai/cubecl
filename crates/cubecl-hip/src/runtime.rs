use crate::{
    HipWmmaCompiler,
    compute::{HipServer, context::HipContext},
    device::AmdDevice,
};
use core::ffi::c_int;
use cubecl_common::{
    device::{Device, DeviceService},
    profile::TimingMethod,
};
use cubecl_core::{
    MemoryConfiguration, Runtime,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        ContiguousElements, DeviceProperties, HardwareProperties, MatrixLayout,
        MemoryDeviceProperties, MmaProperties, TargetProperties, VectorSize, features::Plane,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::{
    ComputeKernel,
    hip::{HipDialect, arch::AMDArchitecture, mma::contiguous_elements_rdna3},
    register_supported_types,
    shared::{
        Architecture, CompilationOptions, CppCompiler, CppSupportedFeatures, DialectWmmaCompiler,
        register_mma_features, register_scaled_mma_features, register_wmma_features,
    },
};
use cubecl_hip_sys::{HIP_SUCCESS, hipDeviceScheduleSpin, hipGetDeviceCount, hipSetDeviceFlags};
use cubecl_runtime::{
    allocator::PitchedMemoryLayoutPolicy, client::ComputeClient, logging::ServerLogger,
};
use std::{ffi::CStr, mem::MaybeUninit, sync::Arc};

/// The values that control how a HIP Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct HipRuntime;

pub type HipCompiler = CppCompiler<HipDialect<HipWmmaCompiler>>;
pub type HipComputeKernel = ComputeKernel<HipDialect<HipWmmaCompiler>>;

impl DeviceService for HipServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let device = AmdDevice::from_id(device_id);

        #[allow(unused_assignments)]
        let mut prop_warp_size = 0;
        #[allow(unused_assignments)]
        let mut prop_arch_name = "";
        #[allow(unused_assignments)]
        let mut prop_max_shared_memory_size = 0;
        #[allow(unused_assignments)]
        let mut max_cube_count = (1, 1, 1);
        #[allow(unused_assignments)]
        let mut prop_max_threads = 0;
        let mut max_cube_dim = (1, 1, 1);
        let mut mem_alignment = 32;
        // SAFETY: Calling HIP FFI to query device properties. The `MaybeUninit` is
        // initialized by `hipGetDevicePropertiesR0600` on success (asserted below), so
        // `assume_init()` is valid. The device index is validated by the `AmdDevice` constructor.
        unsafe {
            let mut ll_device_props = MaybeUninit::uninit();
            let status = cubecl_hip_sys::hipGetDevicePropertiesR0600(
                ll_device_props.as_mut_ptr(),
                device.index as cubecl_hip_sys::hipDevice_t,
            );
            assert_eq!(status, HIP_SUCCESS, "Should get device properties");
            let ll_device_props = ll_device_props.assume_init();
            prop_warp_size = ll_device_props.warpSize;
            prop_arch_name = CStr::from_ptr(ll_device_props.gcnArchName.as_ptr())
                .to_str()
                .unwrap();
            prop_max_shared_memory_size = ll_device_props.sharedMemPerBlock;
            max_cube_count = (
                ll_device_props.maxGridSize[0] as u32,
                ll_device_props.maxGridSize[1] as u32,
                ll_device_props.maxGridSize[2] as u32,
            );
            prop_max_threads = ll_device_props.maxThreadsPerBlock as u32;
            max_cube_dim.0 = ll_device_props.maxThreadsDim[0] as u32;
            max_cube_dim.1 = ll_device_props.maxThreadsDim[1] as u32;
            max_cube_dim.2 = ll_device_props.maxThreadsDim[2] as u32;

            // Just to be sure we check both.
            mem_alignment = usize::max(mem_alignment, ll_device_props.textureAlignment);
            mem_alignment = usize::max(mem_alignment, ll_device_props.surfaceAlignment);
        };
        let normalized_arch_name = prop_arch_name.split(':').next().unwrap_or(prop_arch_name);
        let arch = AMDArchitecture::parse(normalized_arch_name).unwrap();
        assert_eq!(prop_warp_size as u32, arch.warp_size());

        // SAFETY: Calling HIP FFI to set the active device and configure spin-wait scheduling
        // for the current thread. The device index has been validated above by a successful
        // `hipGetDevicePropertiesR0600` call.
        unsafe {
            let status = cubecl_hip_sys::hipSetDevice(device.index as cubecl_hip_sys::hipDevice_t);
            hipSetDeviceFlags(hipDeviceScheduleSpin);

            assert_eq!(
                status, HIP_SUCCESS,
                "Should set the default device for the current thread"
            );
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
            assert_eq!(
                status, HIP_SUCCESS,
                "Should get the available memory of the device"
            );
            total
        };
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_memory as u64 / 4,
            alignment: mem_alignment as u64,
        };

        let supported_wmma_combinations = HipWmmaCompiler::supported_wmma_combinations(&arch);
        let supported_mma_combinations = HipWmmaCompiler::supported_mma_combinations(&arch);
        let supported_scaled_mma_combinations =
            HipWmmaCompiler::supported_scaled_mma_combinations(&arch);

        let topology = HardwareProperties {
            load_width: 128,
            plane_size_min: prop_warp_size as u32,
            plane_size_max: prop_warp_size as u32,
            max_bindings: crate::device::AMD_MAX_BINDINGS,
            max_shared_memory_size: prop_max_shared_memory_size,
            max_cube_count,
            max_units_per_cube: prop_max_threads,
            max_cube_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: if supported_wmma_combinations.is_empty() {
                None
            } else {
                Some(16)
            },
            num_cpu_cores: None,
            max_vector_size: VectorSize::MAX,
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            topology,
            TimingMethod::System,
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

        let comp_opts = CompilationOptions {
            warp_size: arch.warp_size(),
            supports_features: CppSupportedFeatures {
                fast_math: true,
                ..Default::default()
            },
        };
        let hip_ctx = HipContext::new(comp_opts, device_props.clone());
        let logger = Arc::new(ServerLogger::default());
        let policy = PitchedMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
        let utilities = ServerUtilities::new(device_props, logger, (), policy);
        let options = RuntimeOptions::default();

        // SAFETY: `is_integrated_gpu` calls HIP FFI functions with a valid device index.
        let is_integrated = unsafe { is_integrated_gpu(device_id.index_id as i32) };

        HipServer::new(
            hip_ctx,
            mem_properties,
            options.memory_config,
            mem_alignment,
            is_integrated,
            utilities,
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
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
        TargetProperties {
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::RowMajor,
                register_layout_b: MatrixLayout::ColMajor,
                register_layout_acc: MatrixLayout::ColMajor,
                register_duplication_a: 2,
                register_duplication_b: 2,
                register_duplication_acc: 1,
                contiguous_elements: ContiguousElements::new(contiguous_elements_rdna3),
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
            if result == HIP_SUCCESS {
                device_count.try_into().unwrap_or(0)
            } else {
                0
            }
        }
        (0..device_count())
            .map(|i| DeviceId::new(0, i as u32))
            .collect()
    }
}

/// Checks whether the GPU with the given device ID is an integrated (APU) device.
///
/// # Safety
///
/// Calls HIP FFI functions. The caller must ensure `device_id` is a valid HIP device index.
unsafe fn is_integrated_gpu(device_id: i32) -> bool {
    // SAFETY: `hipDeviceProp_tR0600` is a plain-old-data struct; zeroing it is valid.
    let mut props = unsafe { std::mem::zeroed::<cubecl_hip_sys::hipDeviceProp_tR0600>() };
    // SAFETY: `props` is a valid mutable reference and `device_id` is assumed valid by the caller.
    let status = unsafe { cubecl_hip_sys::hipGetDevicePropertiesR0600(&mut props, device_id) };
    if status != HIP_SUCCESS {
        return false; // assume discrete if we can't tell
    }
    props.integrated != 0
}
