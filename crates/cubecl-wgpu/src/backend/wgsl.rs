use cubecl_core::{Compiler, prelude::Visibility, server::KernelArguments};
use cubecl_core::{
    WgpuCompilationOptions,
    ir::{ElemType, UIntKind},
};
use cubecl_ir::{DeviceProperties, Type};
use wgpu::Features;

use crate::WgslCompiler;

pub fn bindings(
    repr: &<WgslCompiler as Compiler>::Representation,
    args: &KernelArguments,
) -> (Vec<Visibility>, Option<Visibility>, bool) {
    let bindings = repr
        .buffers
        .iter()
        .map(|it| {
            if it.item.elem().is_atomic() {
                Visibility::ReadWrite
            } else {
                it.visibility
            }
        })
        .collect::<Vec<_>>();
    let meta = (!args.info.data.is_empty()).then_some(Visibility::Read);
    (bindings, meta, false)
}

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
            // SAFETY: Enabling experimental passthrough shaders.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
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

pub fn register_wgsl_features(
    adapter: &wgpu::Adapter,
    props: &mut cubecl_ir::DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
) {
    register_types(props, adapter);
    if props.supports_type(ElemType::UInt(UIntKind::U64)) {
        comp_options.supports_u64 = true;
    }
}

pub fn register_types(props: &mut DeviceProperties, adapter: &wgpu::Adapter) {
    use cubecl_core::ir::{AddressType, ElemType, FloatKind, IntKind, StorageType};
    use cubecl_ir::features::*;

    props.register_address_type(AddressType::U32);

    let supported_types = [
        ElemType::UInt(UIntKind::U32),
        ElemType::Int(IntKind::I32),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::Flex32),
        ElemType::Bool,
    ];

    let supported_atomic_types = [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all())
    }

    for ty in supported_atomic_types {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ty)),
            AtomicUsage::LoadStore | AtomicUsage::Add,
        );
    }

    let feats = adapter.features();

    if feats.contains(wgpu::Features::SHADER_INT64) {
        props.register_type_usage(ElemType::Int(IntKind::I64), TypeUsage::all());
        props.register_type_usage(ElemType::UInt(UIntKind::U64), TypeUsage::all());
    }
    if feats.contains(wgpu::Features::SHADER_F64) {
        props.register_type_usage(ElemType::Float(FloatKind::F64), TypeUsage::all());
    }
    if feats.contains(wgpu::Features::SHADER_F16) {
        props.register_type_usage(ElemType::Float(FloatKind::F16), TypeUsage::all());
    }
    if feats.contains(wgpu::Features::SHADER_FLOAT32_ATOMIC) {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
            AtomicUsage::LoadStore | AtomicUsage::Add,
        );
    }
}
