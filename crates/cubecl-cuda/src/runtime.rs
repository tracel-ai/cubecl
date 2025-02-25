use std::mem::MaybeUninit;

use cubecl_core::{
    ir::{Elem, FloatKind},
    AtomicFeature, DeviceId, Feature, MemoryConfiguration, Runtime,
};
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
    storage::ComputeStorage,
    ComputeRuntime, DeviceProperties,
};

use crate::{
    compute::{CudaContext, CudaServer, CudaStorage},
    device::CudaDevice,
};
use cubecl_cpp::{
    cuda::{arch::CudaArchitecture, mma::CudaWmmaCompiler},
    register_supported_types,
    shared::register_wmma_features,
    CudaCompiler, WmmaCompiler,
};

/// Options configuring the CUDA runtime.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct CudaRuntime;

type Server = CudaServer;
type Channel = MutexComputeChannel<Server>;

static RUNTIME: ComputeRuntime<CudaDevice, Server, Channel> = ComputeRuntime::new();

fn create_client(device: &CudaDevice, options: RuntimeOptions) -> ComputeClient<Server, Channel> {
    // To get the supported WMMA features, and memory properties, we have to initialize the server immediately.
    cudarc::driver::result::init().unwrap();
    let device_ptr = cudarc::driver::result::device::get(device.index as i32).unwrap();
    let arch = unsafe {
        let major = cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap();
        let minor = cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap();
        major * 10 + minor
    } as u32;
    let arch = CudaArchitecture { version: arch };

    let ctx = unsafe {
        let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
        cudarc::driver::result::ctx::set_current(ctx).unwrap();
        ctx
    };

    let stream = cudarc::driver::result::stream::create(
        cudarc::driver::result::stream::StreamKind::NonBlocking,
    )
    .unwrap();
    let max_memory = unsafe {
        let mut bytes = MaybeUninit::uninit();
        cudarc::driver::sys::lib().cuDeviceTotalMem_v2(bytes.as_mut_ptr(), device_ptr);
        bytes.assume_init() as u64
    };
    let max_shared = unsafe {
        cudarc::driver::result::device::get_attribute(device_ptr, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,).unwrap() as usize
    };
    let storage = CudaStorage::new(stream);
    let mem_properties = MemoryDeviceProperties {
        max_page_size: max_memory / 4,
        alignment: CudaStorage::ALIGNMENT,
    };

    let warp_size = unsafe {
        cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
        .unwrap()
    };
    let hardware_props = HardwareProperties {
        plane_size_min: warp_size as u32,
        plane_size_max: warp_size as u32,
        max_bindings: crate::device::CUDA_MAX_BINDINGS,
        max_shared_memory_size: max_shared,
    };

    let memory_management =
        MemoryManagement::from_configuration(storage, &mem_properties, options.memory_config);

    let mut device_props = DeviceProperties::new(&[Feature::Plane], mem_properties, hardware_props);
    register_supported_types(&mut device_props);
    device_props.register_feature(Feature::Type(Elem::Float(FloatKind::TF32)));
    if arch.version >= 60 {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F64)));
    }
    if arch.version >= 70 {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F16)));
        device_props.register_feature(Feature::Pipeline);
        device_props.register_feature(Feature::Barrier);
    }
    // NOTE: I commented that since I observed synchronisation issues with atomic add for bf16.
    // if arch.version >= 80 {
    //     device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));
    // }
    let supported_wmma_combinations = CudaWmmaCompiler::supported_wmma_combinations(&arch);
    register_wmma_features(supported_wmma_combinations, &mut device_props);

    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));

    let comp_opts = Default::default();
    let cuda_ctx = CudaContext::new(memory_management, comp_opts, stream, ctx, arch);
    let server = CudaServer::new(cuda_ctx);
    ComputeClient::new(MutexComputeChannel::new(server), device_props)
}

impl Runtime for CudaRuntime {
    type Compiler = CudaCompiler;
    type Server = CudaServer;

    type Channel = MutexComputeChannel<CudaServer>;
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            create_client(device, RuntimeOptions::default())
        })
    }

    fn device_id(device: &Self::Device) -> cubecl_core::DeviceId {
        DeviceId::new(0, device.index as u32)
    }

    fn name() -> &'static str {
        "cuda"
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

    fn extension() -> &'static str {
        "cu"
    }
}
