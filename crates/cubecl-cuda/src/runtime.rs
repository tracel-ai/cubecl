use crate::{
    WmmaCompiler,
    compute::{CudaServer, context::CudaContext, valid_strides},
    device::CudaDevice,
};
use cubecl_common::{
    device::{Device, DeviceState},
    profile::TimingMethod,
};
use cubecl_core::{
    CubeCount, CubeDim, MemoryConfiguration, Runtime,
    ir::{
        ElemType, FloatKind, MatrixLayout, MmaProperties, SemanticType, StorageType,
        TargetProperties,
    },
    server::ServerUtilities,
};
use cubecl_cpp::{
    DialectWmmaCompiler,
    cuda::{CudaDialect, arch::CudaArchitecture},
    register_supported_types,
    shared::{
        CompilationOptions, CppCompiler, CppSupportedFeatures, register_mma_features,
        register_scaled_mma_features, register_wmma_features,
    },
};
use cubecl_runtime::{
    DeviceProperties, Plane, Tma, TypeUsage,
    client::ComputeClient,
    logging::ServerLogger,
    memory_management::{HardwareProperties, MemoryDeviceProperties},
};
use cudarc::driver::sys::cuDeviceTotalMem_v2;
use std::{mem::MaybeUninit, sync::Arc};

/// Options configuring the CUDA runtime.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug)]
pub struct CudaRuntime;

impl DeviceState for CudaServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();
        let device = CudaDevice::from_id(device_id);

        // To get the supported WMMA features, and memory properties, we have to initialize the server immediately.
        cudarc::driver::result::init().unwrap();
        let device_id = device.index as i32;
        let device_ptr = cudarc::driver::result::device::get(device_id).unwrap();
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

        // cudamalloc and co. align to _256_ bytes.
        //
        // TODO: Find the correct value from the driver.
        let mem_alignment = 256;

        // Ask the wmma compiler for its supported combinations
        let arch = CudaArchitecture {
            version: arch_version,
        };
        let supported_wmma_combinations = WmmaCompiler::supported_wmma_combinations(&arch);
        let supported_mma_combinations = WmmaCompiler::supported_mma_combinations(&arch);
        let supported_scaled_mma_combinations =
            WmmaCompiler::supported_scaled_mma_combinations(&arch);

        let ctx = unsafe {
            let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
            cudarc::driver::result::ctx::set_current(ctx).unwrap();
            ctx
        };

        let max_memory = unsafe {
            let mut bytes = MaybeUninit::uninit();
            cuDeviceTotalMem_v2(bytes.as_mut_ptr(), device_ptr);
            bytes.assume_init() as u64
        };
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_memory / 4,
            alignment: mem_alignment as u64,
        };

        let mut comp_opts = CompilationOptions {
            supports_features: CppSupportedFeatures {
                fast_math: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let hardware_props = unsafe {
            use cudarc::driver::{result::device::get_attribute, sys::CUdevice_attribute::*};
            let warp_size =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrap() as u32;
            let max_shared = get_attribute(
                device_ptr,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .unwrap() as usize;
            let max_threads = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                .unwrap() as u32;
            let block_dim_x =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X).unwrap();
            let block_dim_y =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y).unwrap();
            let block_dim_z =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z).unwrap();
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

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            hardware_props,
            TimingMethod::System,
        );
        register_supported_types(&mut device_props);
        device_props.register_type_usage(ElemType::Float(FloatKind::TF32), TypeUsage::Conversion);
        if arch_version >= 60 {
            device_props.register_type_usage(
                StorageType::Atomic(ElemType::Float(FloatKind::F64)),
                TypeUsage::AtomicAdd | TypeUsage::AtomicLoadStore,
            );
        }
        if arch_version >= 70 {
            device_props.register_type_usage(
                StorageType::Atomic(ElemType::Float(FloatKind::F16)),
                TypeUsage::AtomicAdd | TypeUsage::AtomicLoadStore,
            );
            device_props.register_semantic_type(SemanticType::Pipeline);
            device_props.register_semantic_type(SemanticType::Barrier);
            device_props.features.plane.insert(Plane::Sync);

            comp_opts.supports_features.grid_constants = true;
        }

        if arch_version >= 75 {
            device_props
                .features
                .ldmatrix
                .insert(ElemType::Float(FloatKind::F16).into());
            device_props
                .features
                .ldmatrix
                .insert(ElemType::Float(FloatKind::BF16).into());
        }

        // NOTE: I commented that since I observed synchronisation issues with atomic add for bf16.
        // if arch.get_version() >= 80 {
        //     device_props.register_feature(Feature::Type(Elem::AtomicFloat(FloatKind::BF16)));
        // }

        if arch_version >= 89 {
            device_props.register_type_usage(
                ElemType::Float(FloatKind::E4M3),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
            device_props.register_type_usage(
                ElemType::Float(FloatKind::E5M2),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
        }
        if arch_version >= 90 {
            device_props.features.tma.insert(Tma::Base);
            device_props.register_semantic_type(SemanticType::TensorMap);
            device_props.features.cube_cluster = true;
            comp_opts.supports_features.clusters = true;
            comp_opts.supports_features.elect_sync = true;
            device_props
                .features
                .stmatrix
                .insert(ElemType::Float(FloatKind::F16).into());
            device_props
                .features
                .stmatrix
                .insert(ElemType::Float(FloatKind::BF16).into());
        }

        if arch_version >= 100 {
            device_props.features.tma.insert(Tma::Im2colWide);
        }

        // NOTE: FP6/FP4 is explicitly not marked as forward compatible, but is compatible within a
        // major version. Try to keep this up to date with new arch major revisions if they also
        // implement it.
        if arch_major == 10 || arch_major == 12 {
            device_props
                .register_type_usage(ElemType::Float(FloatKind::E2M1), TypeUsage::Conversion);
            device_props.register_type_usage(
                StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
            device_props.register_type_usage(
                ElemType::Float(FloatKind::E2M3),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
            device_props.register_type_usage(
                ElemType::Float(FloatKind::E3M2),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
            device_props.register_type_usage(
                ElemType::Float(FloatKind::UE8M0),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
        }

        device_props.features.dynamic_line_size = true;
        device_props.features.alignment = true;
        device_props.features.plane.insert(Plane::Ops);

        register_wmma_features(supported_wmma_combinations, &mut device_props);
        register_mma_features(supported_mma_combinations, &mut device_props);
        register_scaled_mma_features(supported_scaled_mma_combinations, &mut device_props);

        let cuda_ctx = CudaContext::new(comp_opts, ctx, arch);
        let logger = Arc::new(ServerLogger::default());
        let utilities = ServerUtilities::new(device_props, logger, ());

        CudaServer::new(
            cuda_ctx,
            mem_properties,
            options.memory_config,
            mem_alignment,
            device_id,
            utilities,
        )
    }
}

pub type CudaCompiler = CppCompiler<CudaDialect<WmmaCompiler>>;

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
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self::Server>) -> &'static str {
        "cuda"
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
        valid_strides(shape, strides)
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
