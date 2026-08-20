use cubecl_ir::{
    interfaces::TypedExt,
    prelude::*,
    types::{
        ArrayType, AtomicType, Fp8Format, RuntimeArrayType, VectorType,
        scalar::{
            BoolType, Float8E4M3Type, Float8E5M2Type, Float16Type, Float32Type, Float64Type,
            FloatFlex32Type, IndexType,
        },
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

/// WGSL has no 8-bit type, so fp8 only exists packed four lanes to a `u32`; lane 0 is the low
/// byte, which is the byte order of the unpacked vector in memory.
const FP8_LANES_PER_WORD: usize = 4;

fn fp8_unsupported(what: &str) -> ! {
    panic!(
        "fp8 on WGSL is packed {FP8_LANES_PER_WORD} lanes to a u32, a {what} has no representation: \
         convert vectors of {FP8_LANES_PER_WORD}, 8 or 16 lanes and keep them in u32 buffers"
    )
}

macro_rules! fp8_scalar_ty {
    ($ty: ty) => {
        #[type_interface_impl]
        impl TypeToWgsl for $ty {
            fn to_wgsl(&self, _ctx: &Context) -> String {
                fp8_unsupported("scalar")
            }
        }
    };
}

fp8_scalar_ty!(Float8E4M3Type);
fp8_scalar_ty!(Float8E5M2Type);

#[type_interface_impl]
impl TypeToWgsl for VectorType {
    fn to_wgsl(&self, ctx: &Context) -> String {
        if Fp8Format::of_type(ctx, self.inner.scalar_ty(ctx)).is_some() {
            if !self.vectorization.is_multiple_of(FP8_LANES_PER_WORD) {
                fp8_unsupported(&format!("vector of {} lanes", self.vectorization));
            }
            let words = self.vectorization / FP8_LANES_PER_WORD;
            return match words {
                1 => "u32".to_string(),
                words => format!("vec{words}<u32>"),
            };
        }
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
