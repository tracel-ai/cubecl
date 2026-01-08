use cubecl_core::{
    WgpuCompilationOptions,
    ir::{AddressType, UIntKind},
};
use cubecl_cpp::{
    DialectWmmaCompiler,
    metal::{MslDialect, arch::MetalArchitecture},
    shared::register_wmma_features,
};
use cubecl_ir::{
    DeviceProperties,
    features::{EnumSet, Plane, TypeUsage},
};
use wgpu::{
    DeviceDescriptor, Features, Limits,
    hal::{self, Adapter, metal},
};

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
    _adapter: &metal::Adapter,
    props: &mut DeviceProperties,
    _features: Features,
    _comp_options: &mut WgpuCompilationOptions,
) {
    register_types(props);
    register_cmma(props);
    props.features.alignment = true;
    props.features.plane.insert(Plane::Ops);
    props.features.plane.insert(Plane::Sync);
}

fn register_types(props: &mut DeviceProperties) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind, StorageType};

    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let mut register = |elem: StorageType, usage: EnumSet<TypeUsage>| {
        props.register_type_usage(elem, usage);
    };

    let types = [
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
        register(ty.into(), TypeUsage::all_scalar());
    }

    for ty in atomic_types {
        register(
            StorageType::Atomic(ty),
            TypeUsage::AtomicAdd | TypeUsage::AtomicLoadStore,
        )
    }
}

fn register_cmma(props: &mut DeviceProperties) {
    let combinations = MslDialect::supported_wmma_combinations(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
}
