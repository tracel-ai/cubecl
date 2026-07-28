use alloc::string::{String, ToString};
use core::fmt;

use cubecl_common::{e2m1, e4m3, e5m2, flex32, tf32, ue8m0};
use half::{bf16, f16};
use pliron::{
    builtin::{
        type_interfaces::FloatTypeInterface,
        types::{IntegerType, Signedness},
    },
    context::Context,
    derive::{pliron_type, type_interface_impl},
    parsable::{IntoParseResult, ParseResult, StateStream},
    printable,
    utils::apfloat::{self, GetSemantics, Semantics, float_parse, single_to_f32},
};
use rustc_apfloat::ieee::{self, IeeeFloat, NonfiniteBehavior};

use crate::{
    ContextExt, ElemType, FloatKind, IntKind, UIntKind, aligned,
    apfloat::{APFloat, APFloatType, apfloat_type},
    interfaces::{AlignedType, MaybePackedType, ScalarType, SizedType, not_packed},
    scalar, sized,
};

scalar!(IntegerType);
not_packed!(IntegerType);

#[type_interface_impl]
impl AlignedType for IntegerType {
    fn align(&self, _ctx: &Context) -> usize {
        self.width().div_ceil(8) as usize
    }
}

#[type_interface_impl]
impl SizedType for IntegerType {
    fn size(&self, _ctx: &Context) -> usize {
        self.width() as usize / 8
    }
}

#[type_interface_impl]
impl ScalarType for IntegerType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        match (self.width(), self.signedness()) {
            (8, Signedness::Signed) => IntKind::I8.into(),
            (16, Signedness::Signed) => IntKind::I16.into(),
            (32, Signedness::Signed) => IntKind::I32.into(),
            (64, Signedness::Signed) => IntKind::I64.into(),
            (8, _) => UIntKind::U8.into(),
            (16, _) => UIntKind::U16.into(),
            (32, _) => UIntKind::U32.into(),
            (64, _) => UIntKind::U64.into(),
            _ => unreachable!("Unsupported bit width"),
        }
    }
}

#[pliron_type(
    name = "cube.poison",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct PoisonType;

#[pliron_type(
    name = "cube.index",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct IndexType;
scalar!(IndexType);
not_packed!(IndexType);

#[type_interface_impl]
impl AlignedType for IndexType {
    fn align(&self, ctx: &Context) -> usize {
        self.size(ctx)
    }
}

#[type_interface_impl]
impl SizedType for IndexType {
    fn size(&self, ctx: &Context) -> usize {
        ctx.address_type().size()
    }
}

#[type_interface_impl]
impl ScalarType for IndexType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        ElemType::Index
    }
}

macro_rules! float_type {
    ($name: literal, $ty: ident, $kind: ident, $size: literal, $size_bits: expr) => {
        #[pliron_type(name = $name, format = "", generate_get = true, verifier = "succ")]
        #[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
        pub struct $ty;
        scalar!($ty);
        not_packed!($ty);
        aligned!($ty, $size);

        #[type_interface_impl]
        impl ScalarType for $ty {
            fn elem_type(&self, _ctx: &Context) -> ElemType {
                FloatKind::$kind.into()
            }
        }

        #[type_interface_impl]
        impl SizedType for $ty {
            fn size(&self, _ctx: &Context) -> usize {
                $size
            }

            fn size_bits(&self, _ctx: &Context) -> usize {
                $size_bits
            }
        }
    };
    ($name: literal, $ty: ident, $kind: ident, $size: literal) => {
        float_type!($name, $ty, $kind, $size, $size * 8);
    };
}

// Not all floats have semantics that fit the apfloat model, so separate this out
macro_rules! float_semantics {
    ($ty: ty, $semantics: ty) => {
        #[type_interface_impl]
        impl FloatTypeInterface for $ty {
            fn get_semantics(&self) -> Semantics {
                <$semantics>::get_semantics()
            }
        }
    };
}

float_type!("cube.f64", Float64Type, F64, 8);
float_semantics!(Float64Type, apfloat::Double);
apfloat_type!(Float64Type, f64, ieee::DoubleS);

float_type!("cube.f32", Float32Type, F32, 4);
float_semantics!(Float32Type, apfloat::Single);
apfloat_type!(Float32Type, f32, ieee::SingleS);

float_type!("cube.tf32", TFloat32Type, TF32, 4);
apfloat_type!(TFloat32Type, tf32, ieee::SingleS);

float_type!("cube.flex32", FloatFlex32Type, Flex32, 4);
float_semantics!(FloatFlex32Type, apfloat::Single);
apfloat_type!(FloatFlex32Type, flex32, ieee::SingleS);

float_type!("cube.f16", Float16Type, F16, 2);
float_semantics!(Float16Type, apfloat::Half);
apfloat_type!(Float16Type, f16, ieee::HalfS);

float_type!("cube.bf16", BFloat16Type, BF16, 2);
float_semantics!(BFloat16Type, apfloat::BFloat);
apfloat_type!(BFloat16Type, bf16, ieee::BFloatS);

float_type!("cube.ue8m0", Float8E8M0Type, UE8M0, 1);

float_type!("cube.e5m2", Float8E5M2Type, E5M2, 1);
float_semantics!(Float8E5M2Type, apfloat::Float8E5M2);
apfloat_type!(Float8E5M2Type, e5m2, ieee::Float8E5M2S);

float_type!("cube.e4m3", Float8E4M3Type, E4M3, 1);
float_semantics!(Float8E4M3Type, apfloat::Float8E4M3FN);
apfloat_type!(Float8E4M3Type, e4m3, ieee::Float8E4M3FNS);

float_type!("cube.e3m2", Float6E3M2Type, E3M2, 1);

float_type!("cube.e2m3", Float6E2M3Type, E2M3, 1);

float_type!("cube.e2m1", Float4E2M1Type, E2M1, 1, 4);
apfloat_type!(Float4E2M1Type, e2m1, Float4E2M1S);

#[type_interface_impl]
impl FloatTypeInterface for TFloat32Type {
    fn get_semantics(&self) -> Semantics {
        let precision = 11;
        Semantics {
            bits: 19,
            exp_bits: 8,
            precision,
            nonfinite_behavior: NonfiniteBehavior::IEEE754,
            max_exp: 127,
            ieee_max_exp: 127,
            min_exp: -126,
            ieee_min_exp: -126,
            nan_significand_base: 0,
            nan_payload_mask: (1u128 << (precision - 1)) - 1,
            qnan_significand: 1u128 << (precision - 2),
        }
    }
}

pub struct Float6E3M2S;
impl ieee::Semantics for Float6E3M2S {
    const BITS: usize = 6;
    const EXP_BITS: usize = 3;
    const NONFINITE_BEHAVIOR: NonfiniteBehavior = NonfiniteBehavior::NanOnly;
}

pub struct Float6E2M3S;
impl ieee::Semantics for Float6E2M3S {
    const BITS: usize = 6;
    const EXP_BITS: usize = 2;
    const NONFINITE_BEHAVIOR: NonfiniteBehavior = NonfiniteBehavior::NanOnly;
}

pub struct Float4E2M1S;
impl ieee::Semantics for Float4E2M1S {
    const BITS: usize = 4;
    const EXP_BITS: usize = 2;
    const NONFINITE_BEHAVIOR: NonfiniteBehavior = NonfiniteBehavior::NanOnly;
}

#[type_interface_impl]
impl FloatTypeInterface for Float8E8M0Type {
    fn get_semantics(&self) -> Semantics {
        let precision = 1;
        Semantics {
            bits: 8,
            exp_bits: 8,
            precision,
            nonfinite_behavior: NonfiniteBehavior::NanOnly,
            max_exp: 127,
            ieee_max_exp: 127,
            min_exp: -126,
            ieee_min_exp: -126,
            nan_significand_base: 0,
            nan_payload_mask: 0,
            qnan_significand: 0,
        }
    }
}

#[type_interface_impl]
impl FloatTypeInterface for Float6E3M2Type {
    fn get_semantics(&self) -> Semantics {
        IeeeFloat::<Float6E3M2S>::get_semantics()
    }
}

#[type_interface_impl]
impl FloatTypeInterface for Float6E2M3Type {
    fn get_semantics(&self) -> Semantics {
        IeeeFloat::<Float6E2M3S>::get_semantics()
    }
}

#[type_interface_impl]
impl FloatTypeInterface for Float4E2M1Type {
    fn get_semantics(&self) -> Semantics {
        IeeeFloat::<Float4E2M1S>::get_semantics()
    }
}

/// `IeeeFloat::from_bits` assumes the presence of a sign bit and will overflow when extracting this
/// non-existent bit. So we need to do custom conversion here.
#[type_interface_impl]
impl APFloatType for Float8E8M0Type {
    fn value_to_f64(&self, val: APFloat) -> f64 {
        assert!(val.has_semantics::<ue8m0>(), "Should me ue8m0");
        ue8m0::from_bits(val.to_bits() as u8).to_f64()
    }
    fn value_from_f64(&self, val: f64) -> APFloat {
        let bits = ue8m0::from_f64(val).to_bits() as u128;
        APFloat::from_bits::<ue8m0>(bits)
    }
    fn value_to_string(&self, val: APFloat) -> String {
        assert!(val.has_semantics::<ue8m0>(), "Should me ue8m0");
        ue8m0::from_bits(val.to_bits() as u8).to_string()
    }
    fn disp_value(
        &self,
        val: APFloat,
        _: &Context,
        _: &printable::State,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        assert!(val.has_semantics::<ue8m0>(), "Should me ue8m0");
        write!(f, "{}", ue8m0::from_bits(val.to_bits() as u8))
    }
    fn parse_value<'a>(&self, input: &mut StateStream<'a>) -> ParseResult<'a, APFloat> {
        let val = single_to_f32(float_parse::<apfloat::Single>(input, ())?.0);
        let val = APFloat::from_bits::<ue8m0>(ue8m0::from_f32(val).to_bits() as u128);
        Ok(val).into_parse_result()
    }
}

#[pliron_type(
    name = "cube.e2m1x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct Float4E2M1x2Type;
scalar!(Float4E2M1x2Type);
aligned!(Float4E2M1x2Type, 1);
sized!(Float4E2M1x2Type, 1);

#[type_interface_impl]
impl MaybePackedType for Float4E2M1x2Type {
    fn packing_factor(&self, _ctx: &Context) -> usize {
        2
    }
}

#[type_interface_impl]
impl ScalarType for Float4E2M1x2Type {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        FloatKind::E2M1x2.into()
    }
}

#[pliron_type(
    name = "cube.bool",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct BoolType;
aligned!(BoolType, 1);
scalar!(BoolType);
not_packed!(BoolType);

#[type_interface_impl]
impl ScalarType for BoolType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        ElemType::Bool
    }
}
