
use crate::{
    WmmaCompiler,
    compute::{CudaContext, CudaServer, CudaStorage, valid_strides},
    device::CudaDevice,
};
use cubecl_common::profile::TimingMethod;
use cubecl_core::{
    AtomicFeature, CubeCount, CubeDim, Feature, MemoryConfiguration, Runtime, TmaFeature,
    ir::{
        ElemType, FloatKind, IntKind, MatrixLayout, MmaProperties, StorageType, TargetProperties,
        UIntKind,
    },
};
use cubecl_cpp::{
    DialectWmmaCompiler,
    cuda::{CudaDialect, arch::CudaArchitecture},
    register_supported_types,
    shared::{
        CompilationOptions, CppCompiler, register_mma_features, register_scaled_mma_features,
        register_wmma_features,
    },
};
use cubecl_runtime::{
    ComputeRuntime, DeviceProperties,
    channel::MutexComputeChannel,
    client::ComputeClient,
    id::DeviceId,
    memory_management::{HardwareProperties, MemoryDeviceProperties, MemoryManagement},
};
use cudarc::driver::sys::cuDeviceTotalMem_v2;
use cudarc::driver::sys::cuMemGetAllocationGranularity;
use cudarc::driver::sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE;
use cudarc::driver::sys::CUmemLocation;
use cudarc::driver::sys::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED;
use cudarc::driver::sys::CUmemAllocationHandleType_enum::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
use cudarc::driver::sys::CUmemAllocationProp;
use std::mem::MaybeUninit;
use crate::compute::CudaStorageType;
use crate::compute::ExpandableStorage;
/// Options configuring the CUDA runtime.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    expandable_storage_enabled: bool, // ¿Does the runtime want to use expandable storage?
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

    // Check if device supports VMM
    let vmm_supported = unsafe {
        matches!(cudarc::driver::result::device::get_attribute(
       device_ptr,
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
     ), Ok(1))
    };

    // 32 bytes is enough to handle a double4 worth of alignment.
    // NB: cudamalloc and co. actually align to _256_ bytes. Worth
    // trying this in the future to see if it reduces memory coalescing.
    // TODO: Find the correct value from the driver.
    let mem_alignment = 32;


    // Ask the wmma compiler for its supported combinations
    let arch = CudaArchitecture {
        version: arch_version,
    };
    let supported_wmma_combinations = M::supported_wmma_combinations(&arch);
    let supported_mma_combinations = M::supported_mma_combinations(&arch);
    let supported_scaled_mma_combinations = M::supported_scaled_mma_combinations(&arch);

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


  let (mem_properties, storage) = if !vmm_supported && !options.expandable_storage_enabled {

        let storage = CudaStorage::new(mem_alignment, stream);
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_memory / 4,
            alignment: mem_alignment as u64,
        };
       (mem_properties, CudaStorageType::Regular(storage))

   } else {

        let mut granularity: usize = 0;
        let handle_type = {
        #[cfg(unix)]
        {
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        }
        #[cfg(target_os = "windows")]
        {
            CU_MEM_HANDLE_TYPE_WIN32
        }
        };

        let prop = CUmemAllocationProp {
            type_: CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: handle_type,
            location: CUmemLocation {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                id: device_ptr,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: Default::default(),
        };
        unsafe {
            cuMemGetAllocationGranularity(
                &mut granularity,
                &prop,
                cudarc::driver::sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        };
        // For expandable storage, the memory should be aligned to the granularity. Therefore here there is a straightforward way to configure the mem_alignment.
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_memory / 4,
            alignment: granularity as u64,
        };
        let virtual_size =  3 * (max_memory / 4); // Avoid reserving all the mmeory in the gpu
        let storage = ExpandableStorage::new(
            device.index.try_into().unwrap(), // Note that this storage type needs an associated device  to perform allocations. If you do not desire that, the storage API should change, as the [`alloc`] method does not accept a device_id.
            stream,
            virtual_size, // Reserve only a percentage of the gpu memory for safety
            granularity as u64, // Alignment is set to the granularity value.
            granularity as u64, // For now handle size will default to alignment. Can be set configurable in the future.
        );

        (mem_properties, CudaStorageType::Expandable(storage))
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
            CubeCount::new_3d(grid_dim_x as u32, grid_dim_y as u32, grid_dim_z as u32);

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

    let memory_management = MemoryManagement::from_configuration(
        storage,
            &mem_properties,
            options.memory_config,
        );


    let mut device_props = DeviceProperties::new(
        &[Feature::Plane],
        mem_properties,
        hardware_props,
        TimingMethod::System,
    );
    register_supported_types(&mut device_props);
    device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::TF32).into()));
    if arch_version >= 60 {
        device_props.register_feature(Feature::Type(StorageType::Atomic(ElemType::Float(
            FloatKind::F64,
        ))));
    }
    if arch_version >= 70 {
        device_props.register_feature(Feature::Type(StorageType::Atomic(ElemType::Float(
            FloatKind::F16,
        ))));
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
        device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::E4M3).into()));
        device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::E5M2).into()));
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
        device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::E2M1).into()));
        device_props.register_feature(Feature::Type(StorageType::Packed(
            ElemType::Float(FloatKind::E2M1),
            2,
        )));
        device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::E2M3).into()));
        device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::E3M2).into()));
        device_props.register_feature(Feature::Type(ElemType::Float(FloatKind::UE8M0).into()));
    }

    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));

    // Supported by all architectures
    device_props.register_feature(Feature::Type(StorageType::Atomic(ElemType::Int(
        IntKind::I32,
    ))));
    device_props.register_feature(Feature::Type(StorageType::Atomic(ElemType::UInt(
        UIntKind::U32,
    ))));
    device_props.register_feature(Feature::AtomicInt(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicInt(AtomicFeature::Add));
    device_props.register_feature(Feature::AtomicUInt(AtomicFeature::LoadStore));
    device_props.register_feature(Feature::AtomicUInt(AtomicFeature::Add));

    device_props.register_feature(Feature::DynamicLineSize);
    device_props.register_feature(Feature::PlaneOps);

    register_wmma_features(supported_wmma_combinations, &mut device_props);
    register_mma_features(supported_mma_combinations, &mut device_props);
    register_scaled_mma_features(supported_scaled_mma_combinations, &mut device_props);

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
        valid_strides(shape, strides)
    }

    fn device_count() -> usize {
        cudarc::driver::CudaContext::device_count().unwrap_or(0) as usize
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::RowMajor,
                register_layout_b: MatrixLayout::ColMajor,
                register_layout_acc: MatrixLayout::RowMajor,
                register_duplication_a: 1,
                register_duplication_b: 1,
                register_duplication_acc: 1,
            },
        }
    }
}
