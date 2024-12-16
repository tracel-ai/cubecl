use std::{ffi::CStr, mem::MaybeUninit, str::FromStr};

use cubecl_cpp::{
    hip::HipDialect,
    register_supported_types,
    shared::{register_wmma_features, Architecture, CompilationOptions, CppCompiler, WmmaCompiler},
};

use cubecl_core::{Feature, MemoryConfiguration, Runtime};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
    storage::ComputeStorage,
    ComputeRuntime, DeviceProperties,
};

use crate::{
    compute::{HipContext, HipServer, HipStorage},
    device::HipDevice,
    HipWmmaCompiler,
};

/// The values that control how a HIP Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct HipRuntime;

static RUNTIME: ComputeRuntime<HipDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

pub type HipCompiler = CppCompiler<HipDialect<HipWmmaCompiler>>;

type Server = HipServer;
type Channel = MutexComputeChannel<Server>;

fn create_client<M: WmmaCompiler<HipDialect<M>>>(
    device: &HipDevice,
    options: RuntimeOptions,
) -> ComputeClient<Server, Channel> {
    #[allow(unused_assignments)]
    let mut prop_warp_size = 0;
    #[allow(unused_assignments)]
    let mut prop_arch_name = "";
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
    };
    let normalized_arch_name = prop_arch_name.split(':').next().unwrap_or(prop_arch_name);
    let arch = M::Architecture::from_str(normalized_arch_name).unwrap();
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
    let storage = HipStorage::new(stream);
    let mem_properties = MemoryDeviceProperties {
        max_page_size: max_memory as u64 / 4,
        alignment: HipStorage::ALIGNMENT,
    };
    let topology = HardwareProperties {
        plane_size_min: prop_warp_size as u32,
        plane_size_max: prop_warp_size as u32,
        // This is a guess - not clear if ROCM has a limit on the number of bindings,
        // but it's dubious it's more than this.
        max_bindings: 1024,
    };
    let memory_management = MemoryManagement::from_configuration(
        storage,
        mem_properties.clone(),
        options.memory_config,
    );
    let mut device_props = DeviceProperties::new(&[Feature::Plane], mem_properties, topology);
    register_supported_types(&mut device_props);
    let supported_wmma_combinations = M::supported_wmma_combinations(&arch);
    register_wmma_features(supported_wmma_combinations, &mut device_props);

    let comp_opts = CompilationOptions {
        warp_size: arch.warp_size(),
    };
    let hip_ctx = HipContext::new(memory_management, comp_opts, stream);
    let server = HipServer::new(hip_ctx);
    ComputeClient::new(MutexComputeChannel::new(server), device_props)
}

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer;
    type Channel = MutexComputeChannel<HipServer>;
    type Device = HipDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            create_client::<HipWmmaCompiler>(device, RuntimeOptions::default())
        })
    }

    fn name() -> &'static str {
        "hip"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[8, 4, 2]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn extension() -> &'static str {
        "hip"
    }
}
