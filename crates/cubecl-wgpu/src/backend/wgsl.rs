#[cfg(not(all(target_os = "macos", feature = "msl")))]
use cubecl_core::{
    AtomicFeature, Feature, WgpuCompilationOptions,
    ir::{Elem, UIntKind},
};
use cubecl_core::{Compiler, compute::Visibility};
#[cfg(not(all(target_os = "macos", feature = "msl")))]
use cubecl_runtime::DeviceProperties;
#[cfg(not(all(target_os = "macos", feature = "msl")))]
use wgpu::Features;

use crate::WgslCompiler;

pub fn bindings(repr: &<WgslCompiler as Compiler>::Representation) -> Vec<(usize, Visibility)> {
    let mut bindings = repr
        .buffers
        .iter()
        .map(|it| it.visibility)
        .collect::<Vec<_>>();
    if repr.has_metadata {
        bindings.push(Visibility::Read);
    }
    bindings.extend(repr.scalars.iter().map(|_| Visibility::Read));
    bindings.into_iter().enumerate().collect()
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
    if props.feature_enabled(Feature::Type(Elem::UInt(UIntKind::U64))) {
        comp_options.supports_u64 = true;
    }
}

#[cfg(not(all(target_os = "macos", feature = "msl")))]
pub fn register_types(props: &mut DeviceProperties<Feature>, adapter: &wgpu::Adapter) {
    use cubecl_core::ir::{Elem, FloatKind, IntKind};

    let supported_types = [
        Elem::UInt(UIntKind::U32),
        Elem::Int(IntKind::I32),
        Elem::AtomicInt(IntKind::I32),
        Elem::AtomicUInt(UIntKind::U32),
        Elem::Float(FloatKind::F32),
        Elem::Float(FloatKind::Flex32),
        Elem::Bool,
    ];

    let mut register = |ty: Elem| {
        props.register_feature(Feature::Type(ty));
    };

    for ty in supported_types {
        register(ty)
    }

    let feats = adapter.features();

    if feats.contains(wgpu::Features::SHADER_INT64) {
        register(Elem::Int(IntKind::I64));
        register(Elem::UInt(UIntKind::U64));
    }
    if feats.contains(wgpu::Features::SHADER_F64) {
        register(Elem::Float(FloatKind::F64));
    }
    if feats.contains(wgpu::Features::SHADER_F16) {
        register(Elem::Float(FloatKind::F16));
    }
    if feats.contains(wgpu::Features::SHADER_FLOAT32_ATOMIC) {
        register(Elem::AtomicFloat(FloatKind::F32));
        props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
        props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));
    }
}
