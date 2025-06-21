use crate::{
    WmmaCompiler,
    compute::{CudaContext, CudaServer, CudaStorage},
    device::CudaDevice,
};
use cubecl_core::{
    AtomicFeature, CubeDim, Feature, MemoryConfiguration, Runtime, TmaFeature,
    benchmark::TimingMethod,
    ir::{Elem, FloatKind},
};
use cubecl_cpp::{
    DialectWmmaCompiler,
    cuda::{CudaDialect, arch::CudaArchitecture},
    register_supported_types,
    shared::{CompilationOptions, CppCompiler, register_wmma_features},
};
use cubecl_runtime::{
    ComputeRuntime, DeviceProperties,
    channel::MutexComputeChannel,
    client::ComputeClient,
    id::DeviceId,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
};
use cudarc::driver::sys::cuDeviceTotalMem_v2;
use std::mem::MaybeUninit;

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

pub type CudaCompiler = CppCompiler<CudaDialect<WmmaCompiler>>;

fn create_client<M: DialectWmmaCompiler<CudaDialect<M>>>(
    device: &CudaDevice,
    options: RuntimeOptions,
) -> ComputeClient<Server, Channel> {
    // To get the supported WMMA features, and memory properties, we have to initialize the server immediately.
    cudarc::driver::result::init().unwrap();
    let device_ptr = cudarc::driver::result::device::get(device.index as i32).unwrap();
    let arch_major;
    let arch_version = unsafe {
        arch_major = cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .unwrap();
        let minor = cudarc::driver::result::device::get_attribute(
            device_ptr,
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .unwrap();
        arch_major * 10 + minor
    } as u32;
    // 32 bytes is enough to handle a double4 worth of alignment.
    // NB: cudamalloc and co. actually align to _256_ bytes. Worth
    // trying this in the future to see if it reduces memory coalescing.
    //
    // TODO: Find the correct value from the driver.
    let mem_alignment = 32;

    // Ask the wmma compiler for its supported combinations
    let arch = CudaArchitecture {
        version: arch_version,
    };
    let supported_wmma_combinations = M::supported_wmma_combinations(&arch);

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
    let storage = CudaStorage::new(mem_alignment, stream);
    let mem_properties = MemoryDeviceProperties {
        max_page_size: max_memory / 4,
        alignment: mem_alignment as u64,
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
        let num_tensor_cores = tensor_cores_per_sm(arch_version);

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
            min_tensor_cores_dim: if supported_wmma_combinations.is_empty() {
                None
            } else {
                Some(8)
            },
        }
    };

    let memory_management =
        MemoryManagement::from_configuration(storage, &mem_properties, options.memory_config);

    let mut device_props = DeviceProperties::new(
        &[Feature::Plane],
        mem_properties,
        hardware_props,
        TimingMethod::System,
    );
    register_supported_types(&mut device_props);
    device_props.register_feature(Feature::Type(Elem::Float(FloatKind::TF32)));
    if arch_version >= 60 {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F64)));
    }
    if arch_version >= 70 {
        device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::F16)));
        device_props.register_feature(Feature::Pipeline);
        device_props.register_feature(Feature::Barrier);
        device_props.register_feature(Feature::SyncPlane);

        comp_opts.grid_constants = true;
    }

    // NOTE: I commented that since I observed synchronisation issues with atomic add for bf16.
    // if arch.get_version() >= 80 {
    //     device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));
    // }

    if arch_version >= 89 {
        device_props.register_feature(Feature::Type(Elem::Float(FloatKind::E4M3)));
        device_props.register_feature(Feature::Type(Elem::Float(FloatKind::E5M2)));
    }
    if arch_version >= 90 {
        device_props.register_feature(Feature::Tma(TmaFeature::Base));
        device_props.register_feature(Feature::CubeCluster);
        comp_opts.supports_clusters = true;
    }

    if arch_version >= 100 {
        device_props.register_feature(Feature::Tma(TmaFeature::Im2colWide));
    }

    // NOTE: FP6/FP4 is explicitly not marked as forward compatible, but is compatible within a
    // major version. Try to keep this up to date with new arch major revisions if they also
    // implement it.
    if arch_major == 10 || arch_major == 12 {
        device_props.register_feature(Feature::Type(Elem::Float(FloatKind::E2M1)));
        device_props.register_feature(Feature::Type(Elem::Float(FloatKind::E2M3)));
        device_props.register_feature(Feature::Type(Elem::Float(FloatKind::E3M2)));
        device_props.register_feature(Feature::Type(Elem::Float(FloatKind::UE8M0)));
    }

    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));

    device_props.register_feature(Feature::DynamicLineSize);

    register_wmma_features(supported_wmma_combinations, &mut device_props);

    let cuda_ctx = CudaContext::new(memory_management, comp_opts, stream, ctx, arch);
    let server = CudaServer::new(mem_alignment, cuda_ctx);
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
            create_client::<WmmaCompiler>(device, RuntimeOptions::default())
        })
    }

    fn device_id(device: &Self::Device) -> DeviceId {
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

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        let rank = shape.len();
        if strides[rank - 1] != 1 {
            return false;
        }
        if rank <= 1 {
            return true;
        }

        let mut sorted = strides.to_vec();
        sorted.sort();
        sorted.reverse();

        if sorted != strides {
            return false;
        }

        for i in 0..rank - 2 {
            if strides[i] != shape[i + 1] * strides[i + 1] {
                return false;
            }
        }
        true
    }
}
