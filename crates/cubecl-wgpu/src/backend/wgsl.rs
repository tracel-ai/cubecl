#[cfg(not(all(target_os = "macos", feature = "msl")))]
use cubecl_core::{
    AtomicFeature, Feature, WgpuCompilationOptions,
    ir::{ElemType, UIntKind},
};
use cubecl_core::{Compiler, compute::Visibility};
#[cfg(not(all(target_os = "macos", feature = "msl")))]
use cubecl_runtime::DeviceProperties;
#[cfg(not(all(target_os = "macos", feature = "msl")))]
use wgpu::Features;

use crate::WgslCompiler;

pub fn bindings(
    repr: &<WgslCompiler as Compiler>::Representation,
) -> (Vec<Visibility>, Vec<Visibility>) {
    let bindings = repr
        .buffers
        .iter()
        .map(|it| it.visibility)
        .collect::<Vec<_>>();
    let mut meta = vec![];
    if repr.has_metadata {
        meta.push(Visibility::Read);
    }
    meta.extend(repr.scalars.iter().map(|_| Visibility::Read));
    (bindings, meta)
}

#[cfg(not(all(target_os = "macos", feature = "msl")))]
pub async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let limits = adapter.limits();
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: adapter
                .features()
                .difference(Features::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: limits,
            // The default is MemoryHints::Performance, which tries to do some bigger
            // block allocations. However, we already batch allocations, so we
            // can use MemoryHints::MemoryUsage to lower memory usage.
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|err| {
            format!(
                "Unable to request the device with the adapter {:?}, err {:?}",
                adapter.get_info(),
                err
            )
        })
        .unwrap()
}

#[cfg(not(all(target_os = "macos", feature = "msl")))]
pub fn register_wgsl_features(
    adapter: &wgpu::Adapter,
    props: &mut cubecl_runtime::DeviceProperties<cubecl_core::Feature>,
    comp_options: &mut WgpuCompilationOptions,
) {
    register_types(props, adapter);
    if props.feature_enabled(Feature::Type(ElemType::UInt(UIntKind::U64).into())) {
        comp_options.supports_u64 = true;
    }
}

#[cfg(not(all(target_os = "macos", feature = "msl")))]
pub fn register_types(props: &mut DeviceProperties<Feature>, adapter: &wgpu::Adapter) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind, StorageType};

    let supported_types = [
        ElemType::UInt(UIntKind::U32),
        ElemType::Int(IntKind::I32),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::Flex32),
        ElemType::Bool,
    ];

    let supported_atomic_types = [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)];

    let mut register = |ty: StorageType| {
        props.register_feature(Feature::Type(ty));
    };

    for ty in supported_types {
        register(ty.into())
    }

    for ty in supported_atomic_types {
        register(StorageType::Atomic(ty))
    }

    let feats = adapter.features();

    if feats.contains(wgpu::Features::SHADER_INT64) {
        register(ElemType::Int(IntKind::I64).into());
        register(ElemType::UInt(UIntKind::U64).into());
    }
    if feats.contains(wgpu::Features::SHADER_F64) {
        register(ElemType::Float(FloatKind::F64).into());
    }
    if feats.contains(wgpu::Features::SHADER_F16) {
        register(ElemType::Float(FloatKind::F16).into());
    }
    if feats.contains(wgpu::Features::SHADER_FLOAT32_ATOMIC) {
        register(StorageType::Atomic(ElemType::Float(FloatKind::F32)));
        props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
        props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));
    }
}
