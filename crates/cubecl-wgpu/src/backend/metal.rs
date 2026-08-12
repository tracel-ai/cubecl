use cubecl_core::{
    WgpuCompilationOptions,
    ir::{AddressType, UIntKind},
    prelude::Visibility,
    server::KernelArguments,
};
use cubecl_cpp::{
    metal::{arch::MetalArchitecture, supported_cmma_combinations_metal},
    shared::{MslComputeKernel, register_wmma_features},
};
use cubecl_ir::{
    DeviceProperties, Type,
    features::{AtomicUsage, Plane, TypeUsage},
};
use wgpu::{
    DeviceDescriptor, Features, Limits,
    hal::{self, Adapter, metal},
};

pub fn bindings(repr: &MslComputeKernel, args: &KernelArguments) -> (Vec<Visibility>, usize) {
    let buffers = repr.buffers.iter().map(|it| {
        // When slices are shared, it needs to be read-write if ANY of the slices is read-write,
        // and since we can't be sure, we'll assume everything is read-write.
        if cfg!(exclusive_memory_only) {
            *it
        } else {
            Visibility::ReadWrite
        }
    });
    let uniform = args.info.dynamic_metadata_offset >= args.info.data.len();
    let info_vis = (!args.info.data.is_empty()).then_some(match uniform {
        true => Visibility::Uniform,
        false => Visibility::Read,
    });
    (buffers.chain(info_vis).collect(), 0)
}

pub async fn request_metal_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let limits = adapter.limits();
    let features = adapter
        .features()
        .difference(Features::MAPPABLE_PRIMARY_BUFFERS);
    unsafe {
        let hal_adapter = adapter.as_hal::<hal::api::Metal>().unwrap();
        request_device(adapter, &hal_adapter, features, limits)
    }
}

fn request_device(
    wgpu_adapter: &wgpu::Adapter,
    adapter: &metal::Adapter,
    features: Features,
    limits: Limits,
) -> (wgpu::Device, wgpu::Queue) {
    // The default is MemoryHints::Performance, which tries to do some bigger
    // block allocations. However, we already batch allocations, so we
    // can use MemoryHints::MemoryUsage to lower memory usage.
    let memory_hints = wgpu::MemoryHints::MemoryUsage;
    let device = unsafe {
        adapter
            .open(features, &limits, &memory_hints)
            .expect("should create metal HAL device")
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: limits,
        memory_hints,
        trace: wgpu::Trace::Off,
        // SAFETY: Enabling experimental passthrough shaders.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    };

    unsafe {
        wgpu_adapter
            .create_device_from_hal(device, &descriptor)
            .expect("Failed to create wgpu device")
    }
}

pub fn register_metal_features(
    adapter: &wgpu::Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    let features = adapter.features();
    unsafe {
        if let Some(adapter) = adapter.as_hal::<hal::api::Metal>() {
            register_features(&adapter, props, features, comp_options);
        }
    }
}

fn register_features(
    adapter: &metal::Adapter,
    props: &mut DeviceProperties,
    _features: Features,
    _comp_options: &mut WgpuCompilationOptions,
) {
    register_types(props);
    if device_compiles_bfloat(adapter) {
        register_bf16(props);
    }
    register_cmma(props);
    props.features.alignment = true;
    props.features.plane.insert(Plane::Ops);
    props.features.plane.insert(Plane::Sync);
}

fn register_types(props: &mut DeviceProperties) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind};

    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let types = [
        ElemType::Index,
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];

    let atomic_types = [
        ElemType::Int(IntKind::I32),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Float(FloatKind::F32),
    ];

    for ty in types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in atomic_types {
        props
            .register_atomic_type_usage(Type::atomic(ty), AtomicUsage::Add | AtomicUsage::LoadStore)
    }
}

/// Registers `bf16` for the msl passthrough. Only called when
/// [`device_compiles_bfloat`] proved the device+OS pair actually compiles
/// `bfloat` MSL — the codegen for it already exists in `cubecl-cpp`'s metal
/// dialect (reductions fall back through `float`, shuffles through `ushort`),
/// so registration is the only missing link on capable systems.
fn register_bf16(props: &mut DeviceProperties) {
    use cubecl_core::ir::{ElemType, FloatKind};
    props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::all());
}

/// Probe-compiles a one-line `bfloat` kernel on the adapter's raw device.
///
/// `bfloat` needs MSL 3.1 (macOS 14+ / iOS 17+) AND an Apple6+ GPU family —
/// rather than encoding that version/family table here (wgpu-hal keeps its
/// resolved `msl_version` private), the probe tests the real condition
/// directly: can THIS device on THIS OS compile a `bfloat` kernel? A few
/// milliseconds, once per device init, and it fails closed on any error.
fn device_compiles_bfloat(adapter: &metal::Adapter) -> bool {
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice;

    let src = NSString::from_str(concat!(
        "#include <metal_stdlib>\n",
        "using namespace metal;\n",
        "kernel void cubecl_bf16_probe(device bfloat* x [[buffer(0)]],\n",
        "                              uint i [[thread_position_in_grid]]) {\n",
        "    x[i] = bfloat(float(x[i]) + 1.0f);\n",
        "}\n"
    ));
    // Default compile options resolve to the OS's newest supported MSL —
    // the same ceiling wgpu-hal's private `msl_version` ladder tracks.
    adapter
        .raw_device()
        .newLibraryWithSource_options_error(&src, None)
        .is_ok()
}

fn register_cmma(props: &mut DeviceProperties) {
    let combinations = supported_cmma_combinations_metal(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
}
