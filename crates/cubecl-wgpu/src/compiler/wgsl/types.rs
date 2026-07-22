use cubecl_ir::{
    prelude::*,
    types::{
        ArrayType, AtomicType, RuntimeArrayType, VectorType,
        scalar::{BoolType, Float16Type, Float32Type, Float64Type, FloatFlex32Type, IndexType},
    },
};
use pliron::{builtin::types::IntegerType, identifier::Identifier};

use crate::compiler::wgsl::to_wgsl::{TypeExtWgsl, TypeToWgsl};

macro_rules! scalar_ty {
    ($ty: ty, $wgsl: literal) => {
        #[type_interface_impl]
        impl TypeToWgsl for $ty {
            fn to_wgsl(&self, _ctx: &Context) -> String {
                $wgsl.into()
            }
        }
    };
}

scalar_ty!(Float16Type, "f16");
scalar_ty!(Float32Type, "f32");
scalar_ty!(FloatFlex32Type, "f32");
scalar_ty!(Float64Type, "f64");
scalar_ty!(BoolType, "bool");

#[type_interface_impl]
impl TypeToWgsl for IndexType {
    fn to_wgsl(&self, ctx: &Context) -> String {
        ctx.address_type().unsigned_type().to_type(ctx).to_wgsl(ctx)
    }
}

#[type_interface_impl]
impl TypeToWgsl for IntegerType {
    fn to_wgsl(&self, _ctx: &Context) -> String {
        match self.is_signed() {
            true => format!("i{}", self.width()),
            false => format!("u{}", self.width()),
        }
    }
}

#[type_interface_impl]
impl TypeToWgsl for VectorType {
    fn to_wgsl(&self, ctx: &Context) -> String {
        format!("vec{}<{}>", self.vectorization, self.inner.to_wgsl(ctx))
    }
}

#[type_interface_impl]
impl TypeToWgsl for AtomicType {
    fn to_wgsl(&self, ctx: &Context) -> String {
        format!("atomic<{}>", self.inner.to_wgsl(ctx))
    }
}

#[type_interface_impl]
impl TypeToWgsl for ArrayType {
    fn to_wgsl(&self, ctx: &Context) -> String {
        format!("array<{}, {}>", self.inner.to_wgsl(ctx), self.length)
    }
}

#[type_interface_impl]
impl TypeToWgsl for RuntimeArrayType {
    fn to_wgsl(&self, ctx: &Context) -> String {
        format!("array<{}>", self.inner.to_wgsl(ctx))
    }
}

#[pliron_type(
    name = "wgsl.struct",
    format = "$name",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    name: Identifier,
}

#[type_interface_impl]
impl TypeToWgsl for StructType {
    fn to_wgsl(&self, _ctx: &Context) -> String {
        format!("{}", self.name)
    }
}
