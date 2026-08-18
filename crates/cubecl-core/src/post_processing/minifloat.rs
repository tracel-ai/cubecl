//! Software fp8 conversion for targets without hardware fp8.
//!
//! The polyfills work on `u32` bit patterns and use no 8- or 16-bit types, so they also serve
//! backends that cannot store a byte, given the bits in a word. [`LowerMinifloatCast`] rewrites
//! the casts a target cannot convert natively into them.

use cubecl_ir::{
    NamedRewrite, Scope,
    dialect::{base::OperationPtrExt, general::CastOp},
    interfaces::TypedExt,
    prelude::*,
    types::scalar::{Float8E4M3Type, Float8E5M2Type},
};
use pliron::r#type::TypeHandle;

use crate::{self as cubecl, prelude::*};

define_size!(N);

/// An 8-bit float encoding, by its field widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fp8Format {
    /// 4 exponent bits, 3 mantissa bits, bias 7. No infinities; `S.1111.111` is its only NaN.
    E4M3,
    /// 5 exponent bits, 2 mantissa bits, bias 15. IEEE-like: an all-ones exponent is inf or NaN.
    E5M2,
}

impl Fp8Format {
    pub const fn exponent_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 4,
            Fp8Format::E5M2 => 5,
        }
    }

    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    pub const fn bias(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 7,
            Fp8Format::E5M2 => 15,
        }
    }

    /// The largest finite value.
    pub const fn max_value(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }

    /// The code of [`max_value`](Self::max_value), sign bit clear.
    pub const fn max_code(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 0x7E,
            Fp8Format::E5M2 => 0x7B,
        }
    }

    /// The code an encoded NaN takes, sign bit clear.
    pub const fn nan_code(self) -> u32 {
        0x7F
    }

    /// Whether an all-ones exponent means inf or NaN, as in IEEE 754.
    pub const fn has_infinity(self) -> bool {
        matches!(self, Fp8Format::E5M2)
    }

    /// The smallest normal value, `2^(1 - bias)`.
    pub const fn min_normal(self) -> f32 {
        f32::from_bits((127 + 1 - self.bias()) << 23)
    }

    /// The distance between subnormals, `2^(1 - bias - mantissa_bits)`.
    pub const fn subnormal_step(self) -> f32 {
        f32::from_bits((127 + 1 - self.bias() - self.mantissa_bits()) << 23)
    }

    /// The format of a scalar type, if it is one of these.
    pub fn of_type(ctx: &Context, ty: TypeHandle) -> Option<Self> {
        let ty = ty.deref(ctx);
        if ty.is::<Float8E4M3Type>() {
            Some(Fp8Format::E4M3)
        } else if ty.is::<Float8E5M2Type>() {
            Some(Fp8Format::E5M2)
        } else {
            None
        }
    }
}

/// Decodes fp8 bit patterns held in the low byte of each lane. Bits above the byte are ignored.
#[cube]
pub fn fp8_bits_to_f32<N: Size>(bits: Vector<u32, N>, #[comptime] format: Fp8Format) -> Vector<f32, N> {
    let mantissa_bits = comptime![format.mantissa_bits()];
    let exponent_mask = comptime![(1u32 << format.exponent_bits()) - 1];
    let mantissa_mask = comptime![(1u32 << mantissa_bits) - 1];
    let rebias = comptime![127 - format.bias()];
    let mantissa_shift = comptime![23 - mantissa_bits];
    let subnormal_step = comptime![format.subnormal_step()];

    let sign = (bits & Vector::new(0x80u32)) << Vector::new(24u32);
    let exponent = (bits >> Vector::new(mantissa_bits)) & Vector::new(exponent_mask);
    let mantissa = bits & Vector::new(mantissa_mask);

    let normal = sign
        | ((exponent + Vector::new(rebias)) << Vector::new(23u32))
        | (mantissa << Vector::new(mantissa_shift));
    let subnormal = sign
        | Vector::<u32, N>::reinterpret(
            Vector::<f32, N>::cast_from(mantissa) * Vector::new(subnormal_step),
        );
    let value = select_many(exponent.equal(&Vector::new(0u32)), subnormal, normal);

    let nan = sign | Vector::new(0x7FC0_0000u32);
    let result = if comptime![format.has_infinity()] {
        let inf = sign | Vector::new(0x7F80_0000u32);
        let special = select_many(mantissa.equal(&Vector::new(0u32)), inf, nan);
        select_many(exponent.equal(&Vector::new(exponent_mask)), special, value)
    } else {
        let magnitude = bits & Vector::new(0x7Fu32);
        select_many(magnitude.equal(&Vector::new(0x7Fu32)), nan, value)
    };

    Vector::<f32, N>::reinterpret(result)
}

/// Encodes to fp8 bit patterns in the low byte of each lane, upper bits zero.
///
/// Rounds to nearest even. Overflow and infinities saturate to the largest finite value rather
/// than producing NaN or inf, matching the host codecs; a quantization scale must never come back
/// as something that poisons every value it scales.
#[cube]
pub fn f32_to_fp8_bits<N: Size>(value: Vector<f32, N>, #[comptime] format: Fp8Format) -> Vector<u32, N> {
    let mantissa_bits = comptime![format.mantissa_bits()];
    let mantissa_shift = comptime![23 - mantissa_bits];
    let bias = comptime![format.bias()];
    let half_ulp = comptime![1u32 << (mantissa_shift - 1)];
    let subnormal_scale = comptime![1.0 / format.subnormal_step()];
    let min_normal = comptime![format.min_normal()];
    let max_value = comptime![format.max_value()];
    let max_code = comptime![format.max_code()];
    let nan_code = comptime![format.nan_code()];

    let bits = Vector::<u32, N>::reinterpret(value);
    let sign = (bits >> Vector::new(24u32)) & Vector::new(0x80u32);
    let magnitude_bits = bits & Vector::new(0x7FFF_FFFFu32);
    let magnitude = Vector::<f32, N>::reinterpret(magnitude_bits);

    // Below the smallest normal the code is a count of subnormal steps: truncate, then round to
    // nearest even by hand, which needs no float rounding mode and survives fast-math.
    let steps = magnitude * Vector::new(subnormal_scale);
    let truncated = Vector::<u32, N>::cast_from(steps);
    let fraction = steps - Vector::<f32, N>::cast_from(truncated);
    let above_half = fraction.greater_than(&Vector::new(0.5f32));
    let tie_to_odd = fraction
        .equal(&Vector::new(0.5f32))
        .vec_and((truncated & Vector::new(1u32)).equal(&Vector::new(1u32)));
    let round_up = Vector::<u32, N>::cast_from(above_half.or(tie_to_odd));
    let subnormal = truncated + round_up;

    // In the normal range, rebias the exponent and round the mantissa in place; a mantissa that
    // rounds over carries into the exponent on its own.
    let exponent = (magnitude_bits >> Vector::new(23u32)) + Vector::new(bias);
    let mantissa = magnitude_bits & Vector::new(0x007F_FFFFu32);
    let lsb = (mantissa >> Vector::new(mantissa_shift)) & Vector::new(1u32);
    let rounded = ((exponent - Vector::new(127u32)) << Vector::new(23u32) | mantissa)
        + Vector::new(half_ulp - 1)
        + lsb;
    let normal = rounded >> Vector::new(mantissa_shift);

    let code = select_many(magnitude.less_than(&Vector::new(min_normal)), subnormal, normal);
    let code = select_many(
        magnitude.greater_than(&Vector::new(max_value)),
        Vector::new(max_code),
        code,
    );
    let code = select_many(
        magnitude_bits.greater_than(&Vector::new(0x7F80_0000u32)),
        Vector::new(nan_code),
        code,
    );

    code | sign
}

/// Which fp8 formats a target converts in hardware; every other minifloat cast is lowered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeFp8 {
    pub e4m3: bool,
    pub e5m2: bool,
}

impl NativeFp8 {
    pub const NONE: Self = Self {
        e4m3: false,
        e5m2: false,
    };
    pub const ALL: Self = Self {
        e4m3: true,
        e5m2: true,
    };

    pub fn contains(self, format: Fp8Format) -> bool {
        match format {
            Fp8Format::E4M3 => self.e4m3,
            Fp8Format::E5M2 => self.e5m2,
        }
    }
}

pub type LowerMinifloatCastPass = MatchRewritePass<LowerMinifloatCast>;

/// Lowers casts to and from fp8 the target has no instruction for into integer bit manipulation,
/// through `f32`.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerMinifloatCast {
    pub native: NativeFp8,
}

impl LowerMinifloatCast {
    fn emulated(&self, ctx: &Context, value: impl Typed) -> Option<Fp8Format> {
        Fp8Format::of_type(ctx, value.scalar_ty(ctx)).filter(|format| !self.native.contains(*format))
    }
}

impl MatchRewrite for LowerMinifloatCast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<CastOp>(ctx) {
            return false;
        }
        let input = op.operand(ctx, 0);
        let result = op.result(ctx);
        // Bool casts are lowered by the backends themselves and are not what an fp8 buffer holds.
        if input.scalar_ty(ctx).is_bool(ctx) || result.scalar_ty(ctx).is_bool(ctx) {
            return false;
        }
        self.emulated(ctx, input).is_some() || self.emulated(ctx, result).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let input = op.operand(ctx, 0);
        let result_ty = op.result(ctx).get_type(ctx);
        scope.register_size::<N>(input.vector_size(ctx));

        let mut value = input;
        if let Some(format) = self.emulated(ctx, input) {
            value = decode(&scope, value, format);
        }
        let value = match self.emulated(ctx, result_ty) {
            Some(format) => encode(&scope, value, format, result_ty),
            None => cast_value(&scope, value, result_ty),
        };
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

/// fp8 lanes to `f32` lanes: reinterpret as bytes, widen, decode.
fn decode(scope: &Scope, value: Value, format: Fp8Format) -> Value {
    let bytes = reinterpret_value(scope, value, Vector::<u8, N>::__expand_as_type(scope));
    let bits = cast_value(scope, bytes, Vector::<u32, N>::__expand_as_type(scope));
    fp8_bits_to_f32::expand::<N>(scope, bits.into(), format).read_value(scope)
}

/// Any numeric lanes to fp8 lanes: widen to `f32`, encode, narrow to bytes, reinterpret.
fn encode(scope: &Scope, value: Value, format: Fp8Format, result_ty: TypeHandle) -> Value {
    let value = cast_value(scope, value, Vector::<f32, N>::__expand_as_type(scope));
    let bits = f32_to_fp8_bits::expand::<N>(scope, value.into(), format).read_value(scope);
    let bytes = cast_value(scope, bits, Vector::<u8, N>::__expand_as_type(scope));
    reinterpret_value(scope, bytes, result_ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_constants_agree_with_the_codecs() {
        assert_eq!(Fp8Format::E4M3.max_value(), cubecl_common::e4m3::MAX.to_f32());
        assert_eq!(Fp8Format::E5M2.max_value(), cubecl_common::e5m2::MAX.to_f32());
        assert_eq!(
            Fp8Format::E4M3.max_code(),
            cubecl_common::e4m3::MAX.to_bits() as u32
        );
        assert_eq!(
            Fp8Format::E5M2.max_code(),
            cubecl_common::e5m2::MAX.to_bits() as u32
        );
        assert_eq!(
            Fp8Format::E4M3.min_normal(),
            cubecl_common::e4m3::MIN_POSITIVE.to_f32()
        );
        assert_eq!(
            Fp8Format::E5M2.min_normal(),
            cubecl_common::e5m2::MIN_POSITIVE.to_f32()
        );
        assert_eq!(
            Fp8Format::E4M3.subnormal_step(),
            cubecl_common::e4m3::from_bits(1).to_f32()
        );
        assert_eq!(
            Fp8Format::E5M2.subnormal_step(),
            cubecl_common::e5m2::from_bits(1).to_f32()
        );
    }
}
