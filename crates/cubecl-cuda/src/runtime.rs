use std::mem::MaybeUninit;

use cubecl_core::{
    AtomicFeature, CubeDim, DeviceId, Feature, MemoryConfiguration, Runtime, TmaFeature,
    ir::{Elem, FloatKind},
};
use cubecl_runtime::{
    ComputeRuntime, DeviceProperties,
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
    storage::ComputeStorage,
};
use cudarc::driver::sys::cuDeviceTotalMem_v2;

use crate::{
    compute::{CudaContext, CudaServer, CudaStorage},
    device::CudaDevice,
};
use cubecl_cpp::{
    CudaCompiler, DialectWmmaCompiler,
    cuda::{arch::CudaArchitecture, mma::CudaWmmaCompiler},
    register_supported_types,
    shared::{CompilationOptions, register_wmma_features},
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
        cuDeviceTotalMem_v2(bytes.as_mut_ptr(), device_ptr);
        bytes.assume_init() as u64
    };
    let storage = CudaStorage::new(stream);
    let mem_properties = MemoryDeviceProperties {
        max_page_size: max_memory / 4,
        alignment: CudaStorage::ALIGNMENT,
    };

    let mut comp_opts = CompilationOptions::default();

    let hardware_props = unsafe {
        use cudarc::driver::{result::device::get_attribute, sys::CUdevice_attribute::*};
        let warp_size = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrap() as u32;
        let max_shared = get_attribute(
            device_ptr,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
        .unwrap() as usize;
        let max_threads =
            get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap() as u32;
        let block_dim_x = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X).unwrap();
        let block_dim_y = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y).unwrap();
        let block_dim_z = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z).unwrap();
        let max_cube_dim =
            CubeDim::new_3d(block_dim_x as u32, block_dim_y as u32, block_dim_z as u32);

        let grid_dim_x = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X).unwrap();
        let grid_dim_y = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y).unwrap();
        let grid_dim_z = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z).unwrap();
        let max_cube_count =
            CubeDim::new_3d(grid_dim_x as u32, grid_dim_y as u32, grid_dim_z as u32);

        let num_streaming_multiprocessors = Some(
            get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap() as u32,
        );
        let num_tensor_cores = tensor_cores_per_sm(arch.version);

        comp_opts.warp_size = warp_size;

        HardwareProperties {
            plane_size_min: warp_size,
            plane_size_max: warp_size,
            max_bindings: crate::device::CUDA_MAX_BINDINGS,
            max_shared_memory_size: max_shared,
            max_cube_count,
            max_units_per_cube: max_threads,
            max_cube_dim,
            num_streaming_multiprocessors,
            num_tensor_cores,
        }
    };

    let memory_management =
        MemoryManagement::from_configuration(storage, &mem_properties, options.memory_config);

    let mut device_props = DeviceProperties::new(
        &[Feature::Plane],
        mem_properties,
        hardware_props,
        cubecl_runtime::TimeMeasurement::System,
    );
    register_supported_types(&mut device_props);
    device_props.register_feature(Feature::Type(Elem::Float(FloatKind::TF32)));
    if arch.version >= 60 {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F64)));
    }
    if arch.version >= 70 {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F16)));
        device_props.register_feature(Feature::Pipeline);
        device_props.register_feature(Feature::Barrier);

        comp_opts.grid_constants = true;
    }
    if arch.version >= 90 {
        device_props.register_feature(Feature::Tma(TmaFeature::Base));
        device_props.register_feature(Feature::CubeCluster);
        comp_opts.supports_clusters = true;
    }
    if arch.version >= 100 {
        device_props.register_feature(Feature::Tma(TmaFeature::Im2colWide));
    }
    // NOTE: I commented that since I observed synchronisation issues with atomic add for bf16.
    // if arch.version >= 80 {
    //     device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));
    // }
    let supported_wmma_combinations = CudaWmmaCompiler::supported_wmma_combinations(&arch);
    register_wmma_features(supported_wmma_combinations, &mut device_props);

    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));

    device_props.register_feature(Feature::DynamicLineSize);

    let cuda_ctx = CudaContext::new(memory_management, comp_opts, stream, ctx, arch);
    let server = CudaServer::new(cuda_ctx);
    ComputeClient::new(MutexComputeChannel::new(server), device_props, ())
}

fn tensor_cores_per_sm(version: u32) -> Option<u32> {
    match version {
        70 | 75 => Some(8),                           // Volta, Turing
        80 | 86 | 89 | 90 | 91 | 92 | 100 => Some(4), // Ampere, Hopper, Blackwell
        _ => None,                                    // Unknown or unsupported architecture
    }
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

    fn name(_client: &ComputeClient<Self::Server, Self::Channel>) -> &'static str {
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
}
