use alloc::{vec, vec::Vec};

use crate::{
    NoMemoryEffect,
    interfaces::{
        AlignedType, MaybeVectorizedType, MemoryEffect, MemoryEffects, ScalarizableType, TypedExt,
    },
    scalar,
};
use pliron::{
    context::Context,
    derive::{op_interface_impl, type_interface_impl},
    r#type::{TypeHandle, Typed, TypedHandle},
};
use pliron_spirv::{
    ops::{AccessChainOp, InBoundsAccessChainOp, LoadOp},
    spirv::StorageClass,
    types::{FloatType, PointerType, VectorType},
};

NoMemoryEffect!(InBoundsAccessChainOp);
NoMemoryEffect!(AccessChainOp);

#[op_interface_impl]
impl MemoryEffects for LoadOp {
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let ptr = self.get_operand_pointer(ctx);
        let Ok(ptr_ty) = TypedHandle::<PointerType>::from_handle(ptr.get_type(ctx), ctx) else {
            return vec![MemoryEffect::Read(self.get_operand_pointer(ctx))];
        };
        let storage_class = ptr_ty.deref(ctx).storage_class;
        match storage_class {
            // Inherently readonly memory that does not observe writes, memory effects are not
            // relevant to value. Until we add a better way to represent that, treat it as no memory
            // effect.
            StorageClass::UniformConstant
            | StorageClass::Input
            | StorageClass::ShaderRecordBufferKHR
            | StorageClass::IncomingCallableDataKHR
            | StorageClass::IncomingRayPayloadKHR => vec![],
            _ => vec![MemoryEffect::Read(self.get_operand_pointer(ctx))],
        }
    }
}

scalar!(FloatType);

#[type_interface_impl]
impl AlignedType for FloatType {
    fn align(&self, _ctx: &Context) -> usize {
        self.width.div_ceil(8) as usize
    }
}

#[type_interface_impl]
impl AlignedType for VectorType {
    fn align(&self, ctx: &Context) -> usize {
        self.count as usize * self.element_type.align(ctx)
    }
}

#[type_interface_impl]
impl MaybeVectorizedType for VectorType {
    fn vector_size(&self, _ctx: &Context) -> usize {
        self.count as usize
    }
}

#[type_interface_impl]
impl ScalarizableType for VectorType {
    fn scalar_type(&self, ctx: &Context) -> TypeHandle {
        self.element_type.scalar_ty(ctx)
    }
}
