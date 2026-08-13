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

#[cfg(target_vendor = "apple")]
pub fn register_metal_features(
    adapter: &wgpu::Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) -> bool {
    // Canary for ensuring the current Metal language version is 3.2+.
    const LAMBDA_CANARY: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void __canary__(device float* out [[buffer(0)]]) {
            auto f = [](float x) -> float { return x * 2.0; };
            out[0] = f(1.0);
        }
    "#;

    let features = adapter.features();
    unsafe {
        use objc2::rc::autoreleasepool;
        use objc2_foundation::NSString;
        use objc2_metal::{MTLDevice, MTLGPUFamily};

        let Some(adapter) = adapter.as_hal::<hal::api::Metal>() else {
            return false;
        };
        let raw = adapter.raw_device();
        if !raw.supportsFamily(MTLGPUFamily::Apple7) {
            return false;
        };
        let supports_lambdas = autoreleasepool(|_| {
            let canary = NSString::from_str(LAMBDA_CANARY);
            raw.newLibraryWithSource_options_error(&canary, None)
                .is_ok()
        });
        if !supports_lambdas {
            // This is a fixable issue, so we should warn users.
            log::warn!(
                "Device can support native MSL, but Metal compiler version is too old. Upgrading to 3.2 or higher is recommended."
            );
            return false;
        }

        register_features(&adapter, props, features, comp_options);
    }
    true
}

#[cfg(not(target_vendor = "apple"))]
pub fn register_metal_features(
    _: &wgpu::Adapter,
    _: &mut DeviceProperties,
    _: &mut WgpuCompilationOptions,
) -> bool {
    false
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

fn register_cmma(props: &mut DeviceProperties) {
    let combinations = supported_cmma_combinations_metal(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
}
