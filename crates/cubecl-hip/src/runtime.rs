use std::{ffi::CStr, mem::MaybeUninit, str::FromStr};

use cubecl_cpp::{register_supported_types, HipCompiler};

use cubecl_core::{Feature, MemoryConfiguration, Runtime};
use cubecl_hip_sys::HIP_SUCCESS;
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
    ComputeRuntime, DeviceProperties,
};

use crate::{
    arch::AMDArchitecture,
    compute::{HipContext, HipServer, HipStorage},
    device::HipDevice,
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

type Server = HipServer;
type Channel = MutexComputeChannel<Server>;

const MEMORY_OFFSET_ALIGNMENT: u64 = 32;

fn create_client(device: &HipDevice, options: RuntimeOptions) -> ComputeClient<Server, Channel> {
    let mut ctx: cubecl_hip_sys::hipCtx_t = std::ptr::null_mut();

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
    let arch = AMDArchitecture::from_str(prop_arch_name).unwrap();
    assert_eq!(prop_warp_size, arch.warp_size());

    unsafe {
        let status =
            cubecl_hip_sys::hipCtxCreate(&mut ctx, 0, device.index as cubecl_hip_sys::hipDevice_t);
        assert_eq!(status, HIP_SUCCESS, "Should create the HIP context");
    };

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
        alignment: MEMORY_OFFSET_ALIGNMENT,
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
    let hip_ctx = HipContext::new(memory_management, stream, ctx);
    let server = HipServer::new(hip_ctx);
    let mut device_props = DeviceProperties::new(&[Feature::Plane], mem_properties, topology);
    register_supported_types(&mut device_props);
    arch.register_wmma_features(&mut device_props);

    ComputeClient::new(MutexComputeChannel::new(server), device_props)
}

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer;
    type Channel = MutexComputeChannel<HipServer>;
    type Device = HipDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            create_client(device, RuntimeOptions::default())
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

    fn extension() -> &'static str {
        "hip"
    }
}
