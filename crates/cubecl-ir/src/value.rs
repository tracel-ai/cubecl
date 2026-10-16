use core::{fmt::Display, hash::Hash};

use crate::{
    ComplexKind, FloatKind, IntKind, Scope, TypeHash,
    attributes::{BoolAttr, ComplexAttr, FloatAttr, IndexAttr},
    dialect::memory::LoadOp,
    interfaces::TypedExt,
};

use super::{ElemType, Type, UIntKind};
use cubecl_common::{e2m1, e4m3, e5m2, ue8m0};
use derive_more::From;
use float_ord::FloatOrd;
use pliron::{
    attribute::{AttrObj, boxed_attr_cast},
    builtin::{attributes::IntegerAttr, ops::ConstantOp},
    context::Context,
    derive::format,
    r#type::TypedHandle,
    utils::apint::{APInt, bw},
    value::Value,
};

pub fn read_value(scope: &Scope, val: Value) -> Value {
    if val.is_ptr(scope.ctx()) {
        let op = LoadOp::new(scope.ctx_mut(), val);
        scope.register_with_result(&op)
    } else {
        val
    }
}

impl ExpandValue {
    pub fn new(value: Value) -> Self {
        Self::Value(value)
    }

    pub fn constant(value: ConstantValue, ty: impl Into<ElemType>) -> Self {
        let ty = ty.into();
        let value = value.cast_to(ty);
        Self::Constant { value, ty }
    }

    pub fn read_value(&self, scope: &Scope) -> Value {
        let val = self.value(scope);
        read_value(scope, val)
    }

    pub fn value(&self, scope: &Scope) -> Value {
        match self {
            ExpandValue::Value(value) => *value,
            ExpandValue::Constant { value, ty } => {
                let ctx = scope.ctx_mut();
                let value = value.as_attribute(ctx, *ty);
                let value = boxed_attr_cast(value).unwrap();
                let op = ConstantOp::new(scope.ctx_mut(), value);
                scope.register_with_result(&op)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash)]
pub enum ExpandValue {
    Value(Value),
    Constant { value: ConstantValue, ty: ElemType },
}

impl From<Value> for ExpandValue {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, PartialOrd, Ord)]
#[format]
#[repr(u32)]
pub enum Builtin {
    UnitPos,
    UnitPosX,
    UnitPosY,
    UnitPosZ,
    CubePosCluster,
    CubePosClusterX,
    CubePosClusterY,
    CubePosClusterZ,
    CubePos,
    CubePosX,
    CubePosY,
    CubePosZ,
    CubeDim,
    CubeDimX,
    CubeDimY,
    CubeDimZ,
    CubeClusterDim,
    CubeClusterDimX,
    CubeClusterDimY,
    CubeClusterDimZ,
    CubeCount,
    CubeCountX,
    CubeCountY,
    CubeCountZ,
    PlaneDim,
    PlanePos,
    UnitPosPlane,
    AbsolutePos,
    AbsolutePosX,
    AbsolutePosY,
    AbsolutePosZ,
}

/// The scalars are stored with the highest precision possible, but they might get reduced during
/// compilation. For constant propagation, casts are always executed before converting back to the
/// larger type to ensure deterministic output.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, PartialOrd, From)]
#[allow(missing_docs, clippy::derive_ord_xor_partial_ord)]
pub enum ConstantValue {
    Int(i64),
    Float(f64),
    UInt(u64),
    Bool(bool),
    Complex(f64, f64),
}

impl Ord for ConstantValue {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Override float-float comparison with `FloatOrd` since `f64` isn't `Ord`. All other
        // comparisons are safe to unwrap since they're either `Ord` or only compare discriminants.
        match (self, other) {
            (ConstantValue::Float(this), ConstantValue::Float(other)) => {
                FloatOrd(*this).cmp(&FloatOrd(*other))
            }
            (
                ConstantValue::Complex(this_re, this_im),
                ConstantValue::Complex(other_re, other_im),
            ) => FloatOrd(*this_re)
                .cmp(&FloatOrd(*other_re))
                .then_with(|| FloatOrd(*this_im).cmp(&FloatOrd(*other_im))),
            _ => self.partial_cmp(other).unwrap(),
        }
    }
}

impl Eq for ConstantValue {}
impl Hash for ConstantValue {
    fn hash<H: core::hash::Hasher>(&self, ra_expand_state: &mut H) {
        core::mem::discriminant(self).hash(ra_expand_state);
        match self {
            ConstantValue::Int(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Float(f0) => {
                FloatOrd(*f0).hash(ra_expand_state);
            }
            ConstantValue::UInt(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Bool(f0) => {
                f0.hash(ra_expand_state);
            }
            ConstantValue::Complex(re, im) => {
                FloatOrd(*re).hash(ra_expand_state);
                FloatOrd(*im).hash(ra_expand_state);
            }
        }
    }
}

impl ConstantValue {
    /// Returns the value of the constant as a usize.
    ///
    /// It will return [None] if the constant type is a float or a bool.
    pub fn try_as_usize(&self) -> Option<usize> {
        match self {
            ConstantValue::UInt(val) => Some(*val as usize),
            ConstantValue::Int(val) => Some(*val as usize),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) | ConstantValue::Complex(_, _) => None,
        }
    }

    /// Returns the value of the constant as a usize.
    pub fn as_usize(&self) -> usize {
        match self {
            ConstantValue::UInt(val) => *val as usize,
            ConstantValue::Int(val) => *val as usize,
            ConstantValue::Float(val) => *val as usize,
            ConstantValue::Bool(val) => *val as usize,
            ConstantValue::Complex(_, _) => panic!("Complex constants can't be converted to usize"),
        }
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u32(&self) -> Option<u32> {
        self.try_as_u64().map(|it| it as u32)
    }

    /// Returns the value of the scalar as a u32.
    ///
    /// It will panic if the scalar type is a float or a bool.
    pub fn as_u32(&self) -> u32 {
        self.as_u64() as u32
    }

    /// Returns the value of the scalar as a u64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_u64(&self) -> Option<u64> {
        match self {
            ConstantValue::UInt(val) => Some(*val),
            ConstantValue::Int(val) => Some(*val as u64),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) | ConstantValue::Complex(_, _) => None,
        }
    }

    /// Returns the value of the scalar as a u64.
    pub fn as_u64(&self) -> u64 {
        match self {
            ConstantValue::UInt(val) => *val,
            ConstantValue::Int(val) => *val as u64,
            ConstantValue::Float(val) => *val as u64,
            ConstantValue::Bool(val) => *val as u64,
            ConstantValue::Complex(_, _) => panic!("Complex constants can't be converted to u64"),
        }
    }

    /// Returns the value of the scalar as a i64.
    ///
    /// It will return [None] if the scalar type is a float or a bool.
    pub fn try_as_i64(&self) -> Option<i64> {
        match self {
            ConstantValue::UInt(val) => Some(*val as i64),
            ConstantValue::Int(val) => Some(*val),
            ConstantValue::Float(_) => None,
            ConstantValue::Bool(_) | ConstantValue::Complex(_, _) => None,
        }
    }

    /// Returns the value of the scalar as a i128.
    pub fn as_i128(&self) -> i128 {
        match self {
            ConstantValue::UInt(val) => *val as i128,
            ConstantValue::Int(val) => *val as i128,
            ConstantValue::Float(val) => *val as i128,
            ConstantValue::Bool(val) => *val as i128,
            ConstantValue::Complex(_, _) => panic!("Complex constants can't be converted to i128"),
        }
    }

    /// Returns the value of the scalar as a i64.
    pub fn as_i64(&self) -> i64 {
        match self {
            ConstantValue::UInt(val) => *val as i64,
            ConstantValue::Int(val) => *val,
            ConstantValue::Float(val) => *val as i64,
            ConstantValue::Bool(val) => *val as i64,
            ConstantValue::Complex(_, _) => panic!("Complex constants can't be converted to i64"),
        }
    }

    /// Returns the value of the scalar as a i64.
    pub fn as_i32(&self) -> i32 {
        match self {
            ConstantValue::UInt(val) => *val as i32,
            ConstantValue::Int(val) => *val as i32,
            ConstantValue::Float(val) => *val as i32,
            ConstantValue::Bool(val) => *val as i32,
            ConstantValue::Complex(_, _) => panic!("Complex constants can't be converted to i32"),
        }
    }

    /// Returns the value of the scalar as a f64.
    ///
    /// It will return [None] if the scalar type is an int or a bool.
    pub fn try_as_f64(&self) -> Option<f64> {
        match self {
            ConstantValue::Float(val) => Some(*val),
            ConstantValue::Complex(re, _) => Some(*re),
            _ => None,
        }
    }

    /// Returns the value of the scalar as a f64.
    pub fn as_f64(&self) -> f64 {
        match self {
            ConstantValue::UInt(val) => *val as f64,
            ConstantValue::Int(val) => *val as f64,
            ConstantValue::Float(val) => *val,
            ConstantValue::Bool(val) => *val as u8 as f64,
            ConstantValue::Complex(re, _) => *re,
        }
    }

    /// Returns the value of the variable as a bool if it actually is a bool.
    pub fn try_as_bool(&self) -> Option<bool> {
        match self {
            ConstantValue::Bool(val) => Some(*val),
            _ => None,
        }
    }

    /// Returns the scalar's truthiness.
    ///
    /// Complex values are true when either component is nonzero.
    pub fn as_bool(&self) -> bool {
        match self {
            ConstantValue::UInt(val) => *val != 0,
            ConstantValue::Int(val) => *val != 0,
            ConstantValue::Float(val) => *val != 0.,
            ConstantValue::Bool(val) => *val,
            ConstantValue::Complex(re, im) => *re != 0. || *im != 0.,
        }
    }

    pub fn as_attribute(&self, ctx: &Context, elem: ElemType) -> AttrObj {
        let ty = elem.to_type(ctx);
        match self {
            ConstantValue::Int(value) => {
                let value = APInt::from_i64(*value, bw(ty.size_bits(ctx)));
                IntegerAttr::new(TypedHandle::from_handle(ty, ctx).unwrap(), value).into()
            }
            ConstantValue::UInt(value) if elem == ElemType::Index => {
                IndexAttr::new(*value as usize).into()
            }
            ConstantValue::UInt(value) => {
                let value = APInt::from_u64(*value, bw(ty.size_bits(ctx)));
                IntegerAttr::new(TypedHandle::from_handle(ty, ctx).unwrap(), value).into()
            }
            ConstantValue::Float(value) => FloatAttr::from_f64(ctx, ty, *value).into(),
            ConstantValue::Bool(value) => BoolAttr::new(*value).into(),
            ConstantValue::Complex(re, im) => ComplexAttr::from_f64(ctx, ty, *re, *im).into(),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            ConstantValue::Int(val) => *val == 0,
            ConstantValue::Float(val) => *val == 0.0,
            ConstantValue::UInt(val) => *val == 0,
            ConstantValue::Bool(val) => !*val,
            ConstantValue::Complex(re, im) => *re == 0.0 && *im == 0.0,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            ConstantValue::Int(val) => *val == 1,
            ConstantValue::Float(val) => *val == 1.0,
            ConstantValue::UInt(val) => *val == 1,
            ConstantValue::Bool(val) => *val,
            ConstantValue::Complex(re, im) => *re == 1.0 && *im == 0.0,
        }
    }

    pub fn cast_to(&self, other: impl Into<Type>) -> ConstantValue {
        let real = match self {
            ConstantValue::Complex(re, _) => ConstantValue::Float(*re),
            value => *value,
        };

        match other.into().elem_type() {
            ElemType::Index => real.as_u64().into(),
            ElemType::Float(kind) => match kind {
                FloatKind::E2M1 => e2m1::from_f64(real.as_f64()).to_f64(),
                FloatKind::E2M1x2 => e2m1::from_f64(real.as_f64()).to_f64(),
                FloatKind::E2M3 | FloatKind::E3M2 => {
                    unimplemented!("FP6 constants not yet supported")
                }
                FloatKind::E4M3 => e4m3::from_f64(real.as_f64()).to_f64(),
                FloatKind::E5M2 => e5m2::from_f64(real.as_f64()).to_f64(),
                FloatKind::UE8M0 => ue8m0::from_f64(real.as_f64()).to_f64(),
                FloatKind::F16 => half::f16::from_f64(real.as_f64()).to_f64(),
                FloatKind::BF16 => bf16_rounded_once(real).to_f64(),
                FloatKind::Flex32 | FloatKind::TF32 | FloatKind::F32 => real.as_f64() as f32 as f64,
                FloatKind::F64 => real.as_f64(),
            }
            .into(),
            ElemType::Int(kind) => match kind {
                IntKind::I8 => real.as_i64() as i8 as i64,
                IntKind::I16 => real.as_i64() as i16 as i64,
                IntKind::I32 => real.as_i64() as i32 as i64,
                IntKind::I64 => real.as_i64(),
            }
            .into(),
            ElemType::UInt(kind) => match kind {
                UIntKind::U8 => real.as_u64() as u8 as u64,
                UIntKind::U16 => real.as_u64() as u16 as u64,
                UIntKind::U32 => real.as_u64() as u32 as u64,
                UIntKind::U64 => real.as_u64(),
            }
            .into(),
            ElemType::Complex(kind) => {
                let (re, im) = match self {
                    ConstantValue::Complex(re, im) => (*re, *im),
                    value => (value.as_f64(), 0.0),
                };
                match kind {
                    ComplexKind::C32 => ConstantValue::Complex(re as f32 as f64, im as f32 as f64),
                    ComplexKind::C64 => ConstantValue::Complex(re, im),
                }
            }
            ElemType::Bool => self.as_bool().into(),
        }
    }
}

impl Display for ConstantValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConstantValue::Int(val) => write!(f, "{val}"),
            ConstantValue::Float(val) => write!(f, "{val:?}"),
            ConstantValue::UInt(val) => write!(f, "{val}"),
            ConstantValue::Bool(val) => write!(f, "{val}"),
            ConstantValue::Complex(re, im) => write!(f, "({re:?}, {im:?})"),
        }
    }
}

impl ExpandValue {
    pub fn as_const(&self) -> Option<ConstantValue> {
        match self {
            ExpandValue::Constant { value, .. } => Some(*value),
            _ => None,
        }
    }
}

impl Display for ExpandValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExpandValue::Constant { value, ty } => write!(f, "{ty}({value})"),
            ExpandValue::Value(value) => write!(f, "{value:?}"),
        }
    }
}

// Useful with the cube_inline macro.
impl From<&ExpandValue> for ExpandValue {
    fn from(value: &ExpandValue) -> Self {
        *value
    }
}

/// The `bf16` nearest `value`, rounded once as a device conversion rounds it.
/// `half::bf16::from_f64` goes through the nearest `f32` and so rounds twice; rounding to odd at
/// `f32` instead leaves the last rounding exact. This is the host twin of the conversion
/// `cubecl_core::post_processing::bf16` lowers on the device, so a folded constant and the same
/// cast at runtime agree.
fn bf16_rounded_once(value: ConstantValue) -> half::bf16 {
    let odd = match value {
        ConstantValue::Int(value) => {
            let magnitude = u64_to_f32_round_to_odd(value.unsigned_abs());
            if value < 0 { -magnitude } else { magnitude }
        }
        ConstantValue::UInt(value) => u64_to_f32_round_to_odd(value),
        value => f64_to_f32_round_to_odd(value.as_f64()),
    };
    half::bf16::from_f32(odd)
}

/// The nearest `f32`, or when inexact the one beside it toward zero with its low bit set.
fn f64_to_f32_round_to_odd(value: f64) -> f32 {
    let nearest = value as f32;
    if nearest as f64 == value || value.is_nan() {
        return nearest;
    }
    let bits = nearest.to_bits();
    let toward_zero = match (nearest as f64).abs() > value.abs() {
        true => bits - 1,
        false => bits,
    };
    f32::from_bits(toward_zero | 1)
}

/// [`f64_to_f32_round_to_odd`] for an integer magnitude: its top 24 significant bits, with a
/// sticky low bit for whatever was dropped below them.
fn u64_to_f32_round_to_odd(magnitude: u64) -> f32 {
    let significant = u64::BITS - magnitude.leading_zeros();
    let excess = significant.saturating_sub(f32::MANTISSA_DIGITS);
    let kept = magnitude >> excess;
    let odd = kept | u64::from(kept << excess != magnitude);
    let scale = f32::from_bits((excess + (f32::MAX_EXP - 1) as u32) << (f32::MANTISSA_DIGITS - 1));
    odd as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_casts_use_the_real_component_except_for_bool() {
        let value = ConstantValue::Complex(3.5, 2.0);
        assert_eq!(value.cast_to(FloatKind::F64), ConstantValue::Float(3.5));
        assert_eq!(value.cast_to(IntKind::I32), ConstantValue::Int(3));
        assert_eq!(value.cast_to(UIntKind::U32), ConstantValue::UInt(3));
        assert_eq!(
            ConstantValue::Complex(0.0, 1.0).cast_to(ElemType::Bool),
            ConstantValue::Bool(true)
        );
        assert_eq!(
            ConstantValue::Complex(0.0, 0.0).cast_to(ElemType::Bool),
            ConstantValue::Bool(false)
        );
    }

    /// A folded cast to `bf16` rounds once, as the device does: each value sits just past a
    /// `bf16` tie that rounding to `f32` first would land on, then round to even the wrong way.
    #[test]
    fn bf16_constants_round_once() {
        let bf16 = |value: ConstantValue| match value.cast_to(FloatKind::BF16) {
            ConstantValue::Float(value) => half::bf16::from_f64(value).to_bits(),
            other => panic!("a bf16 constant is a float, got {other:?}"),
        };
        // 2^31 + 2^23 + 2^7, and its negation shifted down one.
        assert_eq!(bf16(ConstantValue::UInt(0x8080_0080)), 0x4F01);
        assert_eq!(bf16(ConstantValue::Int(-0x4040_0040)), 0xCE81);
        // 1 + 2^-8 + 2^-52.
        assert_eq!(
            bf16(ConstantValue::Float(f64::from_bits(0x3FF0_1000_0000_0001))),
            0x3F81
        );
        // 2^63 + 2^55 + 1, past what `f64` holds exactly.
        assert_eq!(bf16(ConstantValue::UInt((1 << 63) + (1 << 55) + 1)), 0x5F01);
        assert_eq!(bf16(ConstantValue::Int(i64::MIN)), 0xDF00);
        assert_eq!(bf16(ConstantValue::Float(f64::INFINITY)), 0x7F80);
        assert_eq!(bf16(ConstantValue::Float(-0.0)), 0x8000);
        assert!(half::bf16::from_bits(bf16(ConstantValue::Float(f64::NAN))).is_nan());
    }
}
