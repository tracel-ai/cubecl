use cubecl_ir::prelude::*;
use pliron_spirv::types::PointerType;
use rspirv::spirv::{MemorySemantics, StorageClass};

pub fn semantics_r(ctx: &Context, value: Value) -> MemorySemantics {
    semantics_of(ctx, value) | MemorySemantics::ACQUIRE
}

pub fn semantics_w(ctx: &Context, value: Value) -> MemorySemantics {
    semantics_of(ctx, value) | MemorySemantics::RELEASE
}

pub fn semantics_rw(ctx: &Context, value: Value) -> MemorySemantics {
    semantics_of(ctx, value) | MemorySemantics::ACQUIRE_RELEASE
}

fn semantics_of(ctx: &Context, value: Value) -> MemorySemantics {
    let ty = value.get_type(ctx).deref(ctx);
    if let Some(ptr_ty) = ty.downcast_ref::<PointerType>() {
        match ptr_ty.storage_class {
            StorageClass::StorageBuffer
            | StorageClass::PhysicalStorageBuffer
            | StorageClass::Uniform => MemorySemantics::UNIFORM_MEMORY,
            StorageClass::Workgroup => MemorySemantics::WORKGROUP_MEMORY,
            other => unreachable!("Invalid scope for atomic operation, {other:?}"),
        }
    } else {
        panic!("Should be ptr")
    }
}
