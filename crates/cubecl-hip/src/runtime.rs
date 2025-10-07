use crate::{
    HipWmmaCompiler,
    compute::{HipServer, context::HipContext, contiguous_strides},
    device::AmdDevice,
};
use cubecl_common::profile::TimingMethod;
use cubecl_core::{
    CubeCount, CubeDim, MemoryConfiguration, Runtime, channel,
    ir::{MatrixLayout, MmaProperties, TargetProperties},
};
use cubecl_cpp::{
    hip::{HipDialect, arch::AMDArchitecture},
    register_supported_types,
    shared::{
        Architecture, CompilationOptions, CppCompiler, DialectWmmaCompiler, register_mma_features,
        register_scaled_mma_features, register_wmma_features,
    },
};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    ComputeRuntime, DeviceProperties, Plane,
    client::ComputeClient,
    memory_management::{HardwareProperties, MemoryDeviceProperties},
};
use std::{ffi::CStr, mem::MaybeUninit};

/// The values that control how a HIP Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct HipRuntime;

static RUNTIME: ComputeRuntime<AmdDevice, Server, Channel> = ComputeRuntime::new();

pub type HipCompiler = CppCompiler<HipDialect<HipWmmaCompiler>>;

type Server = HipServer;
type Channel = channel::MutexComputeChannel<Server>;
// type Channel = channel::MpscComputeChannel<Server>;

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
    let mut mem_alignment = 32;
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
        mem_alignment = usize::max(mem_alignment, ll_device_props.textureAlignment);
        mem_alignment = usize::max(mem_alignment, ll_device_props.surfaceAlignment);
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

    let supported_wmma_combinations = M::supported_wmma_combinations(&arch);
    let supported_mma_combinations = M::supported_mma_combinations(&arch);
    let supported_scaled_mma_combinations = M::supported_scaled_mma_combinations(&arch);

    let topology = HardwareProperties {
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

    device_props.features.dynamic_line_size = true;
    device_props.features.plane.insert(Plane::Ops);

    register_wmma_features(supported_wmma_combinations, &mut device_props);
    register_mma_features(supported_mma_combinations, &mut device_props);
    register_scaled_mma_features(supported_scaled_mma_combinations, &mut device_props);

    let comp_opts = CompilationOptions {
        warp_size: arch.warp_size(),
        grid_constants: false,
        supports_clusters: false,
    };
    let hip_ctx = HipContext::new(comp_opts);
    let server = HipServer::new(
        hip_ctx,
        mem_properties,
        options.memory_config,
        mem_alignment,
    );
    ComputeClient::new(Channel::new(server), device_props, ())
}

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer;
    type Channel = Channel;
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
        &[16, 8, 4, 2, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
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
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::ColMajor,
                register_layout_b: MatrixLayout::RowMajor,
                register_layout_acc: MatrixLayout::RowMajor,
                register_duplication_a: 2,
                register_duplication_b: 2,
                register_duplication_acc: 1,
            },
        }
    }
}
