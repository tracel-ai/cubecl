use cubecl_core::{Compiler, Feature, compute::Visibility, ir::UIntKind};
use cubecl_runtime::DeviceProperties;
use wgpu::DeviceDescriptor;

use crate::WgslCompiler;

pub fn bindings(repr: &<WgslCompiler as Compiler>::Representation) -> Vec<(usize, Visibility)> {
    repr.inputs
        .iter()
        .chain(repr.outputs.iter())
        .chain(repr.named.iter().map(|it| &it.1))
        .enumerate()
        .map(|it| (it.0, it.1.visibility))
        .collect()
}

pub async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let limits = adapter.limits();
    adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                required_features: adapter.features(),
                required_limits: limits,
                // The default is MemoryHints::Performance, which tries to do some bigger
                // block allocations. However, we already batch allocations, so we
                // can use MemoryHints::MemoryUsage to lower memory usage.
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
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

pub fn register_types(props: &mut DeviceProperties<Feature>) {
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

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}
