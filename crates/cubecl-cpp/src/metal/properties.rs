use cubecl_core::ir::{
    AddressType, DeviceProperties, ElemType, FloatKind, IntKind, Type, UIntKind,
    features::{AtomicUsage, Plane, TypeUsage},
};

use crate::{
    metal::{
        arch::MetalArchitecture, mma::supported_mma_combinations_metal,
        supported_cmma_combinations_metal,
    },
    shared::{register_mma_features, register_wmma_features},
};

/// What MSL supports, whichever runtime launches it: cubecl's native Metal runtime and wgpu's
/// MSL passthrough compile the same source, so they register the same table.
pub fn register_metal_features(props: &mut DeviceProperties) {
    props.features.alignment = true;
    // MSL's `threadgroup_barrier(mem_flags::mem_device)` orders device memory at device scope.
    props.features.device_memory_scope = true;
    props.features.plane.insert(Plane::Ops);
    props.features.plane.insert(Plane::Sync);
    props.features.plane.insert(Plane::NonUniformControlFlow);

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
        ElemType::Float(FloatKind::BF16),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];

    for ty in types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    // MSL has no fp8 or fp4 type: the emitter stores these as `uint8_t` and converts in software,
    // which is enough to hold them in buffers and cast them, not to compute in them.
    for ty in [
        FloatKind::E4M3,
        FloatKind::E5M2,
        FloatKind::UE8M0,
        FloatKind::E2M1x2,
    ] {
        props.register_type_usage(
            ElemType::Float(ty),
            TypeUsage::Conversion | TypeUsage::Buffer,
        );
        props.register_emulated_conversion(ElemType::Float(ty));
    }

    // MSL's 64-bit atomics exist only on Apple9 and later and only as min and max, so u64
    // is left out: registering it made add, load and store return wrong values.
    for ty in [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)] {
        props.register_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
        props.register_shared_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
    }
    // MSL's float atomics take `device` pointers only: the compiler rejects one in
    // `threadgroup` memory, so shared memory gets none.
    props.register_atomic_type_usage(
        Type::atomic(ElemType::Float(FloatKind::F32)),
        AtomicUsage::Add | AtomicUsage::LoadStore,
    );

    // The `simdgroup_matrix` as a cooperative matrix, and as a manual MMA over each lane's
    // registers.
    let combinations = supported_cmma_combinations_metal(&MetalArchitecture::Metal3);
    register_wmma_features(combinations, props);
    let combinations = supported_mma_combinations_metal(&MetalArchitecture::Metal3);
    register_mma_features(combinations, props);
}
