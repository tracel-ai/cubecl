use crate::{
    compiler::{CudaBackend, CudaCompilationOptions},
    compute::{CudaServer, context::CudaContext},
    device::CudaDevice,
};
use cubecl_common::{
    device::{Device, DeviceService},
    profile::TimingMethod,
};
use cubecl_core::{
    MemoryConfiguration,
    cmma::MatrixLayout,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        ComplexKind, ContiguousElements, DeviceIdentity, DeviceProperties, ElemType, FloatKind,
        HardwareProperties, IntKind, MemoryDeviceProperties, MmaProperties, OpaqueType,
        TargetProperties, Type, UIntKind, VectorSize,
        features::{AtomicUsage, ComplexUsage, Plane, Tma, TypeUsage},
        nvidia::SmArch,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::{
    cuda::{
        self,
        arch::CudaArchitecture,
        mma::{CudaCmmaCompiler, manual::contiguous_elements_cuda},
    },
    register_supported_types,
    shared::{
        CompilationOptions, CppSupportedFeatures, register_mma_features,
        register_scaled_mma_features, register_wmma_features,
    },
};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::{allocator::PitchedMemoryLayoutPolicy, logging::ServerLogger};
use cudarc::driver::sys::{CUDA_VERSION, cuDeviceTotalMem_v2};
use std::{mem::MaybeUninit, sync::Arc};

/// Options configuring the CUDA runtime.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct CudaRuntime;

impl DeviceService for CudaServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();
        let device = CudaDevice::from_id(device_id);

        // To get the supported WMMA features, and memory properties, we have to initialize the server immediately.
        cudarc::driver::result::init().unwrap();
        let device_index = device.index as i32;
        let device_ptr = cudarc::driver::result::device::get(device_index).unwrap();
        let arch_major;
        // SAFETY: Calling CUDA driver FFI to query compute capability attributes.
        // `device_ptr` is a valid device handle obtained from `cudarc::driver::result::device::get`.
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

        // This is the alignment returned by `cuMallocPitched`, so it's the one considered optimal
        // for row alignment by CUDA. This hasn't changed since at least the GTX 700 series.
        // Querying texture row align is a heuristic, but also not guaranteed to be the same.
        let mem_alignment = 512;

        // The name is the only signal for tensor cores. A driver that declines to give one
        // costs only the tensor-core exception, never initialization.
        let device_name = cudarc::driver::result::device::get_name(device_ptr)
            .unwrap_or_else(|_| "unknown CUDA device".to_string());

        // Ask the wmma compiler for its supported combinations
        let arch = CudaArchitecture {
            version: arch_version,
            tensor_cores: CudaArchitecture::has_tensor_cores(arch_version, &device_name),
        };
        let supported_cmma_combinations = CudaCmmaCompiler::Cpp.supported_cmma_combinations(&arch);
        let supported_mma_combinations = cuda::supported_mma_combinations(&arch);
        let supported_scaled_mma_combinations = cuda::supported_scaled_mma_combinations(&arch);

        // SAFETY: `device_ptr` is a valid CUDA device. `primary_ctx::retain` returns the
        // primary context which is then set as current for the calling thread.
        let ctx = unsafe {
            let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
            cudarc::driver::result::ctx::set_current(ctx).unwrap();
            ctx
        };

        // SAFETY: `device_ptr` is valid. `cuDeviceTotalMem_v2` writes the total device memory
        // into the `MaybeUninit`, making `assume_init()` valid on success.
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
                dp4a: arch_version >= 61,
                ..Default::default()
            },
            ..Default::default()
        };

        // SAFETY: `device_ptr` is a valid CUDA device. All `get_attribute` calls query
        // read-only device properties via the CUDA driver API.
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
            let max_cube_dim = (block_dim_x as u32, block_dim_y as u32, block_dim_z as u32);

            let grid_dim_x = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X).unwrap();
            let grid_dim_y = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y).unwrap();
            let grid_dim_z = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z).unwrap();
            let max_cube_count = (grid_dim_x as u32, grid_dim_y as u32, grid_dim_z as u32);

            let num_streaming_multiprocessors = Some(
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap() as u32,
            );
            let num_tensor_cores = tensor_cores_per_sm(&arch);

            comp_opts.warp_size = warp_size as usize;

            HardwareProperties {
                load_width: 128,
                plane_size_min: warp_size,
                plane_size_max: warp_size,
                max_bindings: crate::device::CUDA_MAX_BINDINGS,
                max_shared_memory_size: max_shared,
                max_cube_count,
                max_units_per_cube: max_threads,
                max_cube_dim,
                num_streaming_multiprocessors,
                num_tensor_cores,
                min_tensor_cores_dim: if supported_cmma_combinations.is_empty() {
                    None
                } else {
                    Some(8)
                },
                num_cpu_cores: None,
                last_level_cache_size: None,
                max_vector_size: VectorSize::MAX,
                cube_mma_reserved_shared_memory: 0,
            }
        };

        // The compute capability is what PTX is emitted against, so it is both
        // the compilation namespace and the identity. Built once and shared
        // with `CudaContext` below, so the two cannot disagree.
        let fingerprint = format!("ptx_sm{arch_version}");

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            hardware_props,
            TimingMethod::Device,
            DeviceIdentity {
                name: device_name,
                fingerprint: fingerprint.clone(),
            },
        );
        register_supported_types(&mut device_props);
        for kind in [ComplexKind::C32, ComplexKind::C64] {
            let ty = ElemType::Complex(kind);
            device_props.register_type_usage(ty, TypeUsage::Conversion | TypeUsage::Buffer);
            device_props.register_complex_usage(
                ty,
                ComplexUsage::Core | ComplexUsage::Compare | ComplexUsage::Math,
            );
        }
        device_props.register_type_usage(ElemType::Float(FloatKind::TF32), TypeUsage::Conversion);
        if arch_version >= 60 {
            device_props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F64)),
                AtomicUsage::Add | AtomicUsage::LoadStore | AtomicUsage::Exchange,
            );
        }
        if arch_version >= 70 {
            device_props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F16)),
                AtomicUsage::Add,
            );
            device_props.register_atomic_type_usage(
                Type::atomic(Type::new(ElemType::Float(FloatKind::F16)).with_vector_size(2)),
                AtomicUsage::Add | AtomicUsage::LoadStore | AtomicUsage::Exchange,
            );
            device_props.register_opaque_type(OpaqueType::Barrier);
            device_props.features.plane.insert(Plane::Sync);
            comp_opts.supports_features.grid_constants = true;
        }

        if arch_version >= 75 {
            device_props
                .features
                .matmul
                .ldmatrix
                .insert(ElemType::Float(FloatKind::F16));
            device_props
                .features
                .matmul
                .ldmatrix
                .insert(ElemType::Float(FloatKind::BF16));
            comp_opts.supports_features.fast_tanh = CUDA_VERSION >= 12080;
        }

        if arch_version >= 80 {
            device_props.features.copy_async = true;
        }

        if arch_version >= 90 {
            device_props.features.tma.insert(Tma::Base);
            device_props.register_opaque_type(OpaqueType::TensorMap);
            device_props.features.cube_cluster = true;
            comp_opts.supports_features.clusters = true;
            comp_opts.supports_features.elect_sync = true;
            device_props
                .features
                .matmul
                .stmatrix
                .insert(ElemType::Float(FloatKind::F16));
            device_props
                .features
                .matmul
                .stmatrix
                .insert(ElemType::Float(FloatKind::BF16));

            // bf16 add is only properly supported in sm_90+, even though most bf16 ops are supported
            // earlier. It's technically supported earlier but is missing the now-required `.noftz`
            // modifier, so the behavior is broken.
            for vec in [2, 4, 8] {
                device_props.register_atomic_type_usage(
                    Type::atomic(Type::new(FloatKind::BF16).with_vector_size(vec)),
                    AtomicUsage::Add | AtomicUsage::LoadStore | AtomicUsage::Exchange,
                );
                device_props.register_atomic_type_usage(
                    Type::atomic(Type::new(FloatKind::F16).with_vector_size(vec)),
                    AtomicUsage::Add | AtomicUsage::LoadStore | AtomicUsage::Exchange,
                );
            }
            // PTX docs say min/max is only supported for vectorized f16/bf16, not sure why it's
            // not supported for `f16x2` when it's supported for a vector of `f16` and a vector of
            // `f16x2`. Don't add vectorization of 2 to prevent accidents with optimization code.
            for vec in [4, 8] {
                device_props.register_atomic_type_usage(
                    Type::atomic(Type::new(FloatKind::BF16).with_vector_size(vec)),
                    AtomicUsage::MinMax,
                );
                device_props.register_atomic_type_usage(
                    Type::atomic(Type::new(FloatKind::F16).with_vector_size(vec)),
                    AtomicUsage::MinMax,
                );
            }

            if CUDA_VERSION > 12080 {
                device_props.register_atomic_type_usage(
                    Type::atomic(Type::new(ElemType::Float(FloatKind::F32)).with_vector_size(2)),
                    AtomicUsage::LoadStore | AtomicUsage::Exchange | AtomicUsage::Add,
                );
                device_props.register_atomic_type_usage(
                    Type::atomic(Type::new(ElemType::Float(FloatKind::F32)).with_vector_size(4)),
                    AtomicUsage::LoadStore | AtomicUsage::Exchange | AtomicUsage::Add,
                );
            }
        }

        if arch_version >= 100 {
            device_props.features.tma.insert(Tma::Im2colWide);
            // Breaks swizzle so disable for now and fix in a PR specifically for this
            // if CUDA_VERSION >= 12090 {
            //     device_props.hardware.load_width = 256;
            // }
        }

        // NOTE: FP6/FP4 is explicitly not marked as forward compatible, but is compatible within a
        // major version. Try to keep this up to date with new arch major revisions if they also
        // implement it.
        if arch_major == 10 || arch_major == 11 || arch_major == 12 {
            device_props
                .register_type_usage(ElemType::Float(FloatKind::E2M1), TypeUsage::Conversion);
            device_props.register_type_usage(
                ElemType::Float(FloatKind::E2M1x2),
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

            if CUDA_VERSION >= 12080 {
                device_props.features.tma.insert(Tma::SwizzleAtomicity);
            }
        }

        device_props.features.memory_reinterpret = true;
        device_props.features.alignment = true;
        device_props.features.plane.insert(Plane::Ops);
        device_props
            .features
            .plane
            .insert(Plane::NonUniformControlFlow);

        register_wmma_features(supported_cmma_combinations, &mut device_props);
        register_mma_features(supported_mma_combinations, &mut device_props);
        register_scaled_mma_features(supported_scaled_mma_combinations, &mut device_props);

        // Which backend compiles here decides what may be advertised: the two are not at the
        // same point, and a feature the selected one cannot honour is a kernel that fails to
        // compile rather than a slower one.
        let backend = CudaBackend::default();
        if backend == CudaBackend::Llvm {
            restrict_to_llvm_backend(&mut device_props, &mut comp_opts);
        }

        let comp_opts = CudaCompilationOptions {
            cpp: comp_opts,
            arch: Some(SmArch::new(arch_version, arch.tensor_cores)),
        };
        let cuda_ctx = CudaContext::new(comp_opts, device_props.clone(), ctx, arch, backend);
        let logger = Arc::new(ServerLogger::default());
        let policy = PitchedMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
        let mut utilities = ServerUtilities::new(
            cubecl_common::device::ServiceId::of::<Self>(device_id),
            "cuda",
            device_props,
            CudaRuntime::target_properties(),
            logger,
            policy,
        );
        utilities.server_comm_enabled = true;

        CudaServer::new(
            cuda_ctx,
            mem_properties,
            options.memory_config,
            mem_alignment,
            device_id,
            utilities,
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}


/// Narrows what the device advertises to what the LLVM backend actually lowers.
///
/// The properties above are the C++ backend's, which has had every generation of NVIDIA's
/// hardware features added to it as they shipped. The LLVM backend is at the point of running
/// ordinary kernels: arithmetic, memory, shared memory, the plane operations and the two
/// barriers. Everything it does not lower is taken away here rather than left to fail at
/// compile time, because a consumer picks its algorithm off these properties — cubek's matmul
/// selectors ask for `mma` before they ask anything else — and an advertisement that cannot be
/// honoured is a launch that fails rather than one that falls back.
///
/// Each of these comes back as its lowering lands; see the matrix and TMA work in
/// `cubecl-llvm`'s `nvptx` module.
fn restrict_to_llvm_backend(props: &mut DeviceProperties, comp_opts: &mut CompilationOptions) {
    // Both matrix families are lowered: the cooperative one through `wmma`, the manual one
    // through `mma.sync`. Each is narrowed to the element types its lowering has register
    // shapes for -- `f16` operands throughout, plus the narrow integers on the manual side,
    // which pass their registers as opaque words. `bf16` and `tf32` are in neither for the
    // same reason `bf16` is dropped below: the dialect this backend lowers through has no type
    // for them, so there is nothing to put in a register.
    let half = ElemType::Float(FloatKind::F16);
    let byte = |ty: ElemType| {
        matches!(
            ty,
            ElemType::Int(IntKind::I8) | ElemType::UInt(UIntKind::U8)
        )
    };

    let matmul = &mut props.features.matmul;
    matmul.cmma.retain(|config| {
        config.a_type == half
            && config.b_type == half
            && matches!(
                config.cd_type,
                ElemType::Float(FloatKind::F16) | ElemType::Float(FloatKind::F32)
            )
    });
    matmul.mma.retain(|config| {
        let floats = config.a_type == half
            && config.b_type == half
            && config.cd_type == ElemType::Float(FloatKind::F32);
        // The four signed/unsigned pairings are four instructions over the same registers, so
        // the operands are taken independently.
        let integers = byte(config.a_type)
            && byte(config.b_type)
            && config.cd_type == ElemType::Int(IntKind::I32);
        floats || integers
    });
    // `ldmatrix` and `stmatrix` move 16 bit tiles, whatever those bits stand for, so what they
    // are offered for is whichever operand types survived above.
    matmul.ldmatrix.retain(|elem| *elem == half);
    matmul.stmatrix.retain(|elem| *elem == half);

    // Still on the manual side and still unimplemented: the cube-level API, and the scaled
    // instructions with their `block_scale` operands.
    matmul.cube_mma = Default::default();
    matmul.scaled_mma = Default::default();
    matmul.cmma_tensor_addressing = false;
    if matmul.cmma.is_empty() && matmul.mma.is_empty() {
        props.hardware.num_tensor_cores = None;
        props.hardware.min_tensor_cores_dim = None;
    }

    // No TMA, no clusters, no async copy, and no `mbarrier` behind them.
    props.features.tma = Default::default();
    props.features.cube_cluster = false;
    props.features.copy_async = false;
    props.features.types.opaque.remove(&OpaqueType::TensorMap);
    props.features.types.opaque.remove(&OpaqueType::Barrier);

    // The shuffles go through `shfl.sync` with a full member mask, which requires the plane to
    // be converged. The C++ backend advertises this because its own plane lowering handles a
    // partial mask; until this one does, a diverged plane operation would be undefined rather
    // than merely slow.
    props.features.plane.remove(Plane::NonUniformControlFlow);

    // `bf16` has no type in the LLVM dialect this backend lowers through -- pliron has
    // `builtin.fp16`, `fp32` and `fp64` and nothing between -- so a `bf16` kernel compiles to
    // something that quietly computes zeros. Until it is either given a type or carried as an
    // `i16` the way the minifloats are, it must not be offered.
    let bf16 = ElemType::Float(FloatKind::BF16);
    props.features.types.elem.remove(&bf16);
    props
        .features
        .types
        .atomic
        .retain(|ty, _| ty.elem_type() != bf16);

    // Complex arithmetic is lowered by the C++ backends, not by this one.
    props.features.types.complex.clear();
    for kind in [ComplexKind::C32, ComplexKind::C64] {
        props.features.types.elem.remove(&ElemType::Complex(kind));
    }

    // Vectorized float atomics: the shared atomic lowering handles the scalar widths, and a
    // vector `atomicrmw` is not one instruction on this target.
    props
        .features
        .types
        .atomic
        .retain(|ty, _| ty.vector_size() == 1);

    // Scalars and static metadata ride in a device buffer, not in the kernel's parameter
    // block: the PTX entry ABI presents buffers as pointers in binding order and nothing else,
    // which is the layout `PtxKernelParams` lowers to.
    comp_opts.supports_features.grid_constants = false;
}

fn tensor_cores_per_sm(arch: &CudaArchitecture) -> Option<u32> {
    if !arch.tensor_cores {
        return None;
    }
    match arch.version {
        70 | 75 => Some(8),                           // Volta, Turing
        80 | 86 | 89 | 90 | 91 | 92 | 100 => Some(4), // Ampere, Hopper, Blackwell
        _ => None,                                    // Unknown or unsupported architecture
    }
}

impl Runtime for CudaRuntime {
    type Server = CudaServer;
    type Device = CudaDevice;

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        has_pitched_row_major_strides(shape, strides)
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
                contiguous_elements: ContiguousElements::new(contiguous_elements_cuda),
            },
        }
    }

    fn enumerate_devices(_: u16) -> Vec<cubecl_core::device::DeviceId> {
        let count = cudarc::driver::CudaContext::device_count().unwrap_or(0) as usize;
        (0..count)
            .map(|i| DeviceId {
                type_id: 0,
                index_id: i as u16,
            })
            .collect()
    }
}
