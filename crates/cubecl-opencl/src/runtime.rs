use crate::compute::artifact::OpenClArtifactCompiler;
use crate::compute::programs::OpenClPrograms;
use crate::compute::server::OpenClServer;
use crate::compute::stream::OpenClStreamBackend;
use crate::device::{self, OpenClDevice};
use crate::memory::Memory;
use cubecl_common::device::{Device, DeviceService};
use cubecl_common::profile::TimingMethod;
use cubecl_core::{
    MemoryConfiguration,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        AddressType, DeviceIdentity, DeviceProperties, ElemType, FloatKind, HardwareProperties,
        IntKind, MemoryDeviceProperties, TargetProperties, Type, UIntKind,
        features::{AtomicUsage, MmaConfig, Plane, TypeUsage},
    },
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::opencl::{OpenClCompilationOptions, OpenClOptions};
use cubecl_cpp::shared::{CompilationOptions, register_wmma_features};
use cubecl_server::allocator::ContiguousMemoryLayoutPolicy;
use cubecl_server::compiler::KernelLoader;
use cubecl_server::runtime::Runtime;
use opencl_sys::*;

/// `cl_ext_float_atomics`: the device's single-precision atomic capabilities and the bits this runtime reads.
const CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT: cl_device_info = 0x4231;
const GLOBAL_FP_ATOMIC_ADD: u64 = 1 << 1;
const GLOBAL_FP_ATOMIC_MIN_MAX: u64 = 1 << 2;
const LOCAL_FP_ATOMIC_ADD: u64 = 1 << 17;
const LOCAL_FP_ATOMIC_MIN_MAX: u64 = 1 << 18;

/// Intel's PCI vendor id.
const INTEL: u32 = 0x8086;

/// `OpenCL` runtime for `CubeCL`.
#[derive(Debug, Clone)]
pub struct OpenClRuntime;

impl DeviceService for OpenClServer {
    fn init(device_id: DeviceId) -> Self {
        let requested = OpenClDevice::from_id(device_id);
        let device = *device::devices().get(requested.index).unwrap_or_else(|| {
            panic!(
                "{requested:?}: no such OpenCL device with shared virtual memory and OpenCL C 3.0"
            )
        });
        let context =
            cl3::context::create_context(&[device], std::ptr::null(), None, std::ptr::null_mut())
                .unwrap_or_else(|code| panic!("{requested:?}: clCreateContext failed ({code})"));

        let info = device::Info(device);
        let name = info.string(CL_DEVICE_NAME);
        let intel = info.u64(CL_DEVICE_VENDOR_ID) as u32 == INTEL;
        let plane = plane_size(&info);
        // DPAS, and the block reads and writes its fragments load and store with.
        let dpas = intel
            && plane.is_some()
            && info.extensions(&[
                "cl_intel_subgroup_matrix_multiply_accumulate",
                "cl_intel_subgroups",
                "cl_intel_subgroups_short",
                "cl_intel_subgroups_char",
                "cl_intel_subgroup_local_block_io",
            ]);

        // Pages aligned as the device aligns its buffers' bases (CL_DEVICE_MEM_BASE_ADDR_ALIGN is in
        // bits), within what `clSVMAlloc` takes.
        let alignment = (info.u64(CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8).clamp(16, 128);
        let mem_props =
            MemoryDeviceProperties::new(info.u64(CL_DEVICE_MAX_MEM_ALLOC_SIZE), alignment)
                .with_max_memory(info.u64(CL_DEVICE_GLOBAL_MEM_SIZE));
        let identity = DeviceIdentity {
            fingerprint: format!("opencl_{name}"),
            name: name.clone(),
            physical: None,
        };
        let mut props = DeviceProperties::new(
            Default::default(),
            mem_props.clone(),
            hardware(&info, plane),
            TimingMethod::Device,
            identity,
        );
        register_types(&mut props, &info);
        if let Some(lanes) = plane {
            props.features.plane.insert(Plane::Ops);
            props.features.plane.insert(Plane::Sync);
            if dpas {
                register_wmma_features(dpas_combinations(lanes), &mut props);
            }
        }
        props.features.alignment = true;

        let mut cpp = CompilationOptions {
            warp_size: plane.unwrap_or(32) as usize,
            ..Default::default()
        };
        // Scalars and static metadata by value (`info_st`), the dynamic metadata as a buffer.
        cpp.supports_features.grid_constants = true;
        let options = OpenClCompilationOptions {
            cpp,
            opencl: OpenClOptions {
                subgroup_size: plane,
                dpas,
            },
        };

        let logger = std::sync::Arc::new(cubecl_server::logging::ServerLogger::default());
        let allocator = ContiguousMemoryLayoutPolicy::new(mem_props.alignment as usize);
        let (utilities, captures) = cubecl_core::server::ServerUtilities::init(
            cubecl_common::device::ServiceId::of::<Self>(device_id),
            "opencl",
            props,
            OpenClRuntime::target_properties(),
            logger.clone(),
            allocator,
        );
        // What the cached sources were generated for: the device and driver report what they may
        // use, the options shape them.
        let (platform, driver) = (info.platform(), info.string(CL_DRIVER_VERSION));
        let namespace = format!("{platform}/{name}/{driver}/plane{plane:?}/dpas{dpas}");
        let properties = (*utilities.properties).clone();
        let compiler = OpenClArtifactCompiler::new(context, device, properties, options);
        let kernels = KernelLoader::new(OpenClPrograms::new(compiler, namespace));
        // Intel's driver submits an in-order queue's every command at once: out of order, kept in
        // order by barriers, it batches them.
        let out_of_order = intel
            && info.u64(CL_DEVICE_QUEUE_ON_HOST_PROPERTIES)
                & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                != 0;
        let streams = OpenClStreamBackend::new(
            Memory::of(device, &info),
            context,
            device,
            out_of_order,
            mem_props,
            MemoryConfiguration::default(),
            logger,
            captures,
        );
        OpenClServer::new(context, kernels, streams, std::sync::Arc::new(utilities))
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities.clone() as ServerUtilitiesHandle
    }
}

/// The size every kernel using planes is compiled for (`cl_intel_required_subgroup_size`): the
/// smallest the device runs (a power of two: some drivers list sizes no kernel runs at), which DPAS
/// takes (8 on Xe-LPG / HPG, 16 on Xe2 / HPC). None without every plane operation the dialect
/// lowers: subgroups (an `OpenCL` C 3.0 feature or extension), and the ballot, shuffles, scans of
/// any operation and votes.
fn plane_size(info: &device::Info) -> Option<u32> {
    let subgroups =
        info.extensions(&["cl_khr_subgroups"]) || info.has_feature("__opencl_c_subgroups");
    let operations = info.extensions(&[
        "cl_intel_required_subgroup_size",
        "cl_khr_subgroup_ballot",
        "cl_khr_subgroup_shuffle",
        "cl_khr_subgroup_shuffle_relative",
        "cl_khr_subgroup_non_uniform_arithmetic",
        "cl_khr_subgroup_non_uniform_vote",
    ]);
    info.sizes(CL_DEVICE_SUB_GROUP_SIZES_INTEL as cl_device_info)
        .into_iter()
        .filter(|&size| (4..=128).contains(&size) && size.is_power_of_two())
        .min()
        .filter(|_| subgroups && operations)
        .map(|size| size as u32)
}

fn hardware(info: &device::Info, plane: Option<u32>) -> HardwareProperties {
    let dims = info.sizes(CL_DEVICE_MAX_WORK_ITEM_SIZES);
    let dim = |i: usize| dims.get(i).copied().unwrap_or(1) as u32;
    // Intel counts each EU as a compute unit: the cores are its slices' sub-slices (0 elsewhere).
    let slices =
        info.u64(CL_DEVICE_NUM_SLICES_INTEL) * info.u64(CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL);
    let cores = match slices > 0 {
        true => slices,
        false => info.u64(CL_DEVICE_MAX_COMPUTE_UNITS),
    };
    HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        plane_size_min: plane.unwrap_or(32),
        plane_size_max: plane.unwrap_or(32),
        // Kernel arguments: a pointer per binding, the dynamic metadata's buffer, and the info
        // struct by value (up to 512 bytes).
        max_bindings: (info
            .u64(CL_DEVICE_MAX_PARAMETER_SIZE)
            .saturating_sub(512 + 8)
            / 8)
        .clamp(8, 64) as u32,
        max_shared_memory_size: info.u64(CL_DEVICE_LOCAL_MEM_SIZE) as usize,
        max_cube_count: (u32::MAX, u16::MAX as u32, u16::MAX as u32),
        max_units_per_cube: info.u64(CL_DEVICE_MAX_WORK_GROUP_SIZE).min(u32::MAX as u64) as u32,
        max_cube_dim: (dim(0), dim(1), dim(2)),
        num_streaming_multiprocessors: Some(cores as u32),
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        num_cpu_cores: None,
        last_level_cache_size: None,
        max_vector_size: 4,
        cube_mma_reserved_shared_memory: 0,
    }
}

/// DPAS tiles: 8 rows, a subgroup wide, K = 16 for f16 inputs and 32 for 8-bit ones.
fn dpas_combinations(lanes: u32) -> Vec<MmaConfig> {
    let f16 = ElemType::Float(FloatKind::F16);
    let (i8, u8) = (ElemType::Int(IntKind::I8), ElemType::UInt(UIntKind::U8));
    let i32 = ElemType::Int(IntKind::I32);
    let config = |a_type, b_type, cd_type, k| MmaConfig {
        a_type,
        b_type,
        cd_type,
        m: 8,
        n: lanes,
        k,
    };
    vec![
        config(f16, f16, ElemType::Float(FloatKind::F32), 16),
        config(f16, f16, f16, 16),
        config(i8, i8, i32, 32),
        config(u8, u8, i32, 32),
        config(u8, i8, i32, 32),
        config(i8, u8, i32, 32),
    ]
}

fn register_types(props: &mut DeviceProperties, info: &device::Info) {
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);
    let int64 = info.has_feature("__opencl_c_int64");
    let mut types = vec![
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];
    if int64 {
        types.extend([ElemType::UInt(UIntKind::U64), ElemType::Int(IntKind::I64)]);
    }
    if info.extensions(&["cl_khr_fp16"]) {
        types.push(ElemType::Float(FloatKind::F16));
    }
    for ty in types {
        props.register_type_usage(ty, TypeUsage::all());
    }
    // Atomics without a scope argument are device-scoped, an optional `OpenCL` C 3.0 feature.
    let device_scope =
        info.u64(CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES) & CL_DEVICE_ATOMIC_SCOPE_DEVICE != 0;
    props.features.device_memory_scope = device_scope;
    if device_scope {
        register_atomics(props, info, int64);
    }
}

fn register_atomics(props: &mut DeviceProperties, info: &device::Info, int64: bool) {
    let mut integers = vec![ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)];
    if int64 && info.extensions(&["cl_khr_int64_base_atomics", "cl_khr_int64_extended_atomics"]) {
        integers.extend([ElemType::Int(IntKind::I64), ElemType::UInt(UIntKind::U64)]);
    }
    for ty in integers {
        props.register_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
        props.register_shared_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
    }
    // Float add / min / max per address space, from `cl_ext_float_atomics`' capability bits.
    let float = match info.extensions(&["cl_ext_float_atomics"]) {
        true => info.u64(CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT),
        false => 0,
    };
    let usage = |add: u64, min_max: u64| {
        let mut usage = AtomicUsage::LoadStore | AtomicUsage::Exchange;
        for (bits, op) in [(add, AtomicUsage::Add), (min_max, AtomicUsage::MinMax)] {
            if float & bits != 0 {
                usage |= op;
            }
        }
        usage
    };
    let f32_atomic = Type::atomic(ElemType::Float(FloatKind::F32));
    let global = usage(GLOBAL_FP_ATOMIC_ADD, GLOBAL_FP_ATOMIC_MIN_MAX);
    props.register_atomic_type_usage(f32_atomic, global);
    let local = usage(LOCAL_FP_ATOMIC_ADD, LOCAL_FP_ATOMIC_MIN_MAX);
    props.register_shared_atomic_type_usage(f32_atomic, local);
}

impl Runtime for OpenClRuntime {
    type Server = OpenClServer;
    type Device = OpenClDevice;

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }

    fn enumerate_devices(type_id: u16) -> Vec<DeviceId> {
        let count = match type_id {
            0 => device::devices().len(),
            _ => 0,
        };
        (0..count)
            .map(|index| OpenClDevice { index }.to_id())
            .collect()
    }

    /// A GPU: a machine whose only `OpenCL` device is a CPU implementation (`PoCL`) leaves the
    /// default to a native CPU runtime.
    fn is_available() -> bool {
        device::devices()
            .first()
            .is_some_and(|&device| device::Info(device).is_gpu())
    }
}
