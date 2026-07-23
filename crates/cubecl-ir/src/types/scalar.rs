use pliron::{
    builtin::{
        type_interfaces::FloatTypeInterface,
        types::{IntegerType, Signedness},
    },
    context::Context,
    derive::{pliron_type, type_interface_impl},
    utils::apfloat::{self, GetSemantics, Semantics},
};
use rustc_apfloat::ieee::NonfiniteBehavior;

use crate::{
    ElemType, FloatKind, IntKind, UIntKind, aligned,
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
// May need only align_of::<u32>() for 32-bit addressing, but it's safe to give it more
aligned!(IndexType, align_of::<u64>());
sized!(IndexType, size_of::<u64>());
scalar!(IndexType);
not_packed!(IndexType);

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

float_type!("cube.f32", Float32Type, F32, 4);
float_semantics!(Float32Type, apfloat::Single);

float_type!("cube.tf32", TFloat32Type, TF32, 4);

float_type!("cube.flex32", FloatFlex32Type, Flex32, 4);
float_semantics!(FloatFlex32Type, apfloat::Single);

float_type!("cube.f16", Float16Type, F16, 2);
float_semantics!(Float16Type, apfloat::Half);

float_type!("cube.bf16", BFloat16Type, BF16, 2);
float_semantics!(BFloat16Type, apfloat::BFloat);

float_type!("cube.ue8m0", Float8E8M0Type, UE8M0, 1);

float_type!("cube.e5m2", Float8E5M2Type, E5M2, 1);
float_semantics!(Float8E5M2Type, apfloat::Float8E5M2);

float_type!("cube.e4m3", Float8E4M3Type, E4M3, 1);
float_semantics!(Float8E4M3Type, apfloat::Float8E4M3FN);

float_type!("cube.e3m2", Float6E3M2Type, E3M2, 1);

float_type!("cube.e2m3", Float6E2M3Type, E2M3, 1);

float_type!("cube.e2m1", Float4E2M1Type, E2M1, 1, 4);

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
            nan_significand_base: (1u128 << (precision - 1)) - 1,
            nan_payload_mask: 0,
            qnan_significand: 0,
        }
    }
}

#[type_interface_impl]
impl FloatTypeInterface for Float6E3M2Type {
    fn get_semantics(&self) -> Semantics {
        let precision = 3;
        Semantics {
            bits: 6,
            exp_bits: 3,
            precision,
            nonfinite_behavior: NonfiniteBehavior::NanOnly,
            max_exp: 3,
            ieee_max_exp: 3,
            min_exp: -2,
            ieee_min_exp: -2,
            nan_significand_base: (1u128 << (precision - 1)) - 1,
            nan_payload_mask: 0,
            qnan_significand: 0,
        }
    }
}

#[type_interface_impl]
impl FloatTypeInterface for Float6E2M3Type {
    fn get_semantics(&self) -> Semantics {
        let precision = 4;
        Semantics {
            bits: 6,
            exp_bits: 2,
            precision,
            nonfinite_behavior: NonfiniteBehavior::NanOnly,
            max_exp: 2,
            ieee_max_exp: 2,
            min_exp: -1,
            ieee_min_exp: -1,
            nan_significand_base: (1u128 << (precision - 1)) - 1,
            nan_payload_mask: 0,
            qnan_significand: 0,
        }
    }
}

#[type_interface_impl]
impl FloatTypeInterface for Float4E2M1Type {
    fn get_semantics(&self) -> Semantics {
        let precision = 2;
        Semantics {
            bits: 4,
            exp_bits: 2,
            precision,
            nonfinite_behavior: NonfiniteBehavior::NanOnly,
            max_exp: 2,
            ieee_max_exp: 2,
            min_exp: -1,
            ieee_min_exp: -1,
            nan_significand_base: (1u128 << (precision - 1)) - 1,
            nan_payload_mask: 0,
            qnan_significand: 0,
        }
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
