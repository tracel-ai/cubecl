use cubecl_ir::{
    interfaces::{HasElementType, TypedExt},
    prelude::*,
    rewrite::visit_all_values,
    types::{
        ArrayType, AtomicType, Fp8Format, RuntimeArrayType, VectorType,
        scalar::{
            BoolType, Float8E4M3Type, Float8E5M2Type, Float16Type, Float32Type, Float64Type,
            FloatFlex32Type, IndexType,
        },
    },
};
use pliron::{
    builtin::types::IntegerType, identifier::Identifier, input_err_noloc, operation::Operation,
    result::Result, r#type::type_cast,
};
use thiserror::Error;

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

#[derive(Debug, Error)]
#[error(
    "fp8 on WGSL is packed {FP8_LANES_PER_WORD} lanes to a u32, a {0} has no representation: use \
     vectors of {FP8_LANES_PER_WORD}, 8 or 16 lanes"
)]
pub struct Fp8Unsupported(String);

/// Rejects every fp8 value WGSL cannot lay out, before any pass runs on it. Both the cast
/// lowering and the type printer below would otherwise hit the same rule with a panic, which the
/// device thread swallows into a warning and hands the caller a zeroed buffer.
pub fn check_fp8_lanes(ctx: &Context, module: Ptr<Operation>) -> Result<()> {
    let mut bad = None;
    visit_all_values(ctx, &mut bad, module, |ctx, bad, value| {
        if bad.is_none()
            && let Some(lanes) = fp8_lanes(ctx, value.get_type(ctx))
            && !lanes.is_multiple_of(FP8_LANES_PER_WORD)
        {
            *bad = Some(lanes);
        }
    });
    match bad {
        Some(lanes) => input_err_noloc!(Fp8Unsupported(describe(lanes))),
        None => Ok(()),
    }
}

/// The fp8 lane count `ty` holds, looking through pointers and arrays. `None` when no fp8 is
/// involved, `Some(1)` for a bare scalar.
fn fp8_lanes(ctx: &Context, ty: TypeHandle) -> Option<usize> {
    let deref = ty.deref(ctx);
    if let Some(vector) = deref.downcast_ref::<VectorType>() {
        return Fp8Format::of_type(ctx, vector.inner).map(|_| vector.vectorization);
    }
    if Fp8Format::of_type(ctx, ty).is_some() {
        return Some(1);
    }
    // Scalars are their own element type, so recursing on one would not terminate.
    let elem = type_cast::<dyn HasElementType>(&*deref)?.element_type(ctx)?;
    (elem != ty).then(|| fp8_lanes(ctx, elem)).flatten()
}

fn describe(lanes: usize) -> String {
    match lanes {
        1 => "scalar".to_string(),
        lanes => format!("vector of {lanes} lanes"),
    }
}

fn fp8_unsupported(what: &str) -> ! {
    panic!("{}", Fp8Unsupported(what.to_string()))
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
                fp8_unsupported(&describe(self.vectorization));
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
