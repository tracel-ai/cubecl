use std::{ffi::CStr, mem::MaybeUninit};

use cubecl_cpp::{
    hip::{HipDialect, arch::AMDArchitecture},
    register_supported_types,
    shared::{
        Architecture, CompilationOptions, CppCompiler, DialectWmmaCompiler, register_wmma_features,
    },
};

use cubecl_common::profile::TimingMethod;
use cubecl_core::{
    AtomicFeature, CubeCount, CubeDim, Feature, MemoryConfiguration, Runtime,
    ir::{Elem, FloatKind, IntKind, UIntKind},
};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::id::DeviceId;
use cubecl_runtime::{
    ComputeRuntime, DeviceProperties,
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
};

use crate::{
    HipWmmaCompiler,
    compute::{HipContext, HipServer, HipStorage, contiguous_strides},
    device::AmdDevice,
};

/// The values that control how a HIP Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct HipRuntime;

static RUNTIME: ComputeRuntime<AmdDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

pub type HipCompiler = CppCompiler<HipDialect<HipWmmaCompiler>>;

type Server = HipServer;
type Channel = MutexComputeChannel<Server>;

fn create_client<M: DialectWmmaCompiler<HipDialect<M>>>(
    device: &AmdDevice,
    options: RuntimeOptions,
) -> ComputeClient<Server, Channel> {
    #[allow(unused_assignments)]
    let mut prop_warp_size = 0;
    #[allow(unused_assignments)]
    let mut prop_arch_name = "";
    #[allow(unused_assignments)]
    let mut prop_max_shared_memory_size = 0;
    #[allow(unused_assignments)]
    let mut max_cube_count = CubeCount::new_single();
    #[allow(unused_assignments)]
    let mut prop_max_threads = 0;
    let mut max_cube_dim = CubeDim::new_single();
    let mut mem_aligment = 32;
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
        max_cube_count = CubeCount::new_3d(
            ll_device_props.maxGridSize[0] as u32,
            ll_device_props.maxGridSize[1] as u32,
            ll_device_props.maxGridSize[2] as u32,
        );
        prop_max_threads = ll_device_props.maxThreadsPerBlock as u32;
        max_cube_dim.x = ll_device_props.maxThreadsDim[0] as u32;
        max_cube_dim.y = ll_device_props.maxThreadsDim[1] as u32;
        max_cube_dim.z = ll_device_props.maxThreadsDim[2] as u32;

        // Just to be sure we check both.
        mem_aligment = usize::max(mem_aligment, ll_device_props.textureAlignment);
        mem_aligment = usize::max(mem_aligment, ll_device_props.surfaceAlignment);
    };
    let normalized_arch_name = prop_arch_name.split(':').next().unwrap_or(prop_arch_name);
    let arch = AMDArchitecture::parse(normalized_arch_name).unwrap();
    assert_eq!(prop_warp_size as u32, arch.warp_size());

    unsafe {
        let status = cubecl_hip_sys::hipSetDevice(device.index as cubecl_hip_sys::hipDevice_t);
        assert_eq!(
            status, HIP_SUCCESS,
            "Should set the default device for the current thread"
        );
    }

    let stream = unsafe {
        let mut stream: cubecl_hip_sys::hipStream_t = std::ptr::null_mut();
        let stream_status = cubecl_hip_sys::hipStreamCreate(&mut stream);
        assert_eq!(stream_status, HIP_SUCCESS, "Should create a stream");
        stream
    };

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
    let storage = HipStorage::new(mem_aligment, stream);
    let mem_properties = MemoryDeviceProperties {
        max_page_size: max_memory as u64 / 4,
        alignment: mem_aligment as u64,
    };
    let supported_wmma_combinations = M::supported_wmma_combinations(&arch);
    let topology = HardwareProperties {
        plane_size_min: prop_warp_size as u32,
        plane_size_max: prop_warp_size as u32,
        // This is a guess - not clear if ROCM has a limit on the number of bindings,
        // but it's dubious it's more than this.
        max_bindings: 1024,
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
    };
    let memory_management =
        MemoryManagement::from_configuration(storage, &mem_properties, options.memory_config);
    let mut device_props = DeviceProperties::new(
        &[Feature::Plane],
        mem_properties,
        topology,
        TimingMethod::System,
    );
    register_supported_types(&mut device_props);
    // Not sure if there's a good way to check for support on HIP
    device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F32)));
    // TODO look into unsafeAtomicAdd (https://github.com/ROCm/HIP/issues/3573120)
    // device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F16)));
    // device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));

    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));

    // Supported by all architectures
    device_props.register_feature(Feature::Type(Elem::AtomicInt(IntKind::I32)));
    device_props.register_feature(Feature::Type(Elem::AtomicUInt(UIntKind::U32)));
    device_props.register_feature(Feature::AtomicInt(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicInt(AtomicFeature::Add));
    device_props.register_feature(Feature::AtomicUInt(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicUInt(AtomicFeature::Add));

    device_props.register_feature(Feature::DynamicLineSize);

    register_wmma_features(supported_wmma_combinations, &mut device_props);

    let comp_opts = CompilationOptions {
        warp_size: arch.warp_size(),
        grid_constants: false,
        supports_clusters: false,
    };
    let hip_ctx = HipContext::new(memory_management, comp_opts, stream);
    let server = HipServer::new(mem_aligment, hip_ctx);
    ComputeClient::new(MutexComputeChannel::new(server), device_props, ())
}

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer;
    type Channel = MutexComputeChannel<HipServer>;
    type Device = AmdDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            create_client::<HipWmmaCompiler>(device, RuntimeOptions::default())
        })
    }

    fn name(_client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str {
        "hip"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[8, 4, 2, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn device_id(device: &Self::Device) -> DeviceId {
        DeviceId::new(0, device.index as u32)
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
