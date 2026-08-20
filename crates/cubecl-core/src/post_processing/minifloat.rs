//! Software fp8 conversion, on `u32` bit patterns only so that a backend with no 8- or 16-bit types
//! can still call it on the bits in a word.

use cubecl_ir::{
    NamedRewrite, Scope,
    dialect::{base::OperationPtrExt, general::CastOp},
    interfaces::TypedExt,
    prelude::*,
    types::Fp8Format,
};
use enumset::EnumSet;
use pliron::r#type::TypeHandle;

use crate::{self as cubecl, prelude::*};

define_size!(N);

const F32_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
const F32_MANTISSA_MASK: u32 = (1 << F32_MANTISSA_BITS) - 1;
const F32_MAGNITUDE_MASK: u32 = u32::MAX >> 1;
const F32_EXPONENT_BIAS: u32 = (f32::MAX_EXP - 1) as u32;
const F32_INFINITY_BITS: u32 = f32::INFINITY.to_bits();
const F32_NAN_BITS: u32 = f32::NAN.to_bits();
const FP8_SIGN_BIT: u32 = 1 << (u8::BITS - 1);
const FP8_MAGNITUDE_MASK: u32 = FP8_SIGN_BIT - 1;
const SIGN_SHIFT: u32 = u32::BITS - u8::BITS;

/// Bits above the low byte are ignored.
#[cube]
pub fn fp8_bits_to_f32<N: Size>(
    bits: Vector<u32, N>,
    #[comptime] format: Fp8Format,
) -> Vector<f32, N> {
    let mantissa_bits = comptime![format.mantissa_bits()];
    let exponent_mask = comptime![(1u32 << format.exponent_bits()) - 1];
    let mantissa_mask = comptime![(1u32 << mantissa_bits) - 1];
    let rebias = comptime![F32_EXPONENT_BIAS - format.bias()];
    let mantissa_shift = comptime![F32_MANTISSA_BITS - mantissa_bits];
    let subnormal_step = comptime![format.subnormal_step()];

    let sign = (bits & Vector::new(FP8_SIGN_BIT)) << Vector::new(SIGN_SHIFT);
    let exponent = (bits >> Vector::new(mantissa_bits)) & Vector::new(exponent_mask);
    let mantissa = bits & Vector::new(mantissa_mask);

    let normal = sign
        | ((exponent + Vector::new(rebias)) << Vector::new(F32_MANTISSA_BITS))
        | (mantissa << Vector::new(mantissa_shift));
    let subnormal = sign
        | Vector::<u32, N>::reinterpret(
            Vector::<f32, N>::cast_from(mantissa) * Vector::new(subnormal_step),
        );
    let value = select_many(exponent.equal(&Vector::new(0u32)), subnormal, normal);

    let nan = sign | Vector::new(F32_NAN_BITS);
    let result = if comptime![format.has_infinity()] {
        let inf = sign | Vector::new(F32_INFINITY_BITS);
        let special = select_many(mantissa.equal(&Vector::new(0u32)), inf, nan);
        select_many(exponent.equal(&Vector::new(exponent_mask)), special, value)
    } else {
        let magnitude = bits & Vector::new(FP8_MAGNITUDE_MASK);
        select_many(
            magnitude.equal(&Vector::new(FP8_MAGNITUDE_MASK)),
            nan,
            value,
        )
    };

    Vector::<f32, N>::reinterpret(result)
}

/// Round to nearest even; overflow and infinities saturate to the largest finite value, as the host
/// codecs do.
#[cube]
pub fn f32_to_fp8_bits<N: Size>(
    value: Vector<f32, N>,
    #[comptime] format: Fp8Format,
) -> Vector<u32, N> {
    let mantissa_bits = comptime![format.mantissa_bits()];
    let mantissa_shift = comptime![F32_MANTISSA_BITS - mantissa_bits];
    let rebias = comptime![format.bias().wrapping_sub(F32_EXPONENT_BIAS)];
    let half_ulp = comptime![1u32 << (mantissa_shift - 1)];
    let subnormal_scale = comptime![1.0 / format.subnormal_step()];
    let min_normal = comptime![format.min_normal()];
    let max_value = comptime![format.max_value()];
    let max_code = comptime![format.max_code()];
    let nan_code = comptime![format.nan_code()];

    let bits = Vector::<u32, N>::reinterpret(value);
    let sign = (bits >> Vector::new(SIGN_SHIFT)) & Vector::new(FP8_SIGN_BIT);
    let magnitude_bits = bits & Vector::new(F32_MAGNITUDE_MASK);
    let magnitude = Vector::<f32, N>::reinterpret(magnitude_bits);

    // Rounding by hand: the usual magic-number trick does not survive fast-math reassociation.
    let steps = magnitude * Vector::new(subnormal_scale);
    let truncated = Vector::<u32, N>::cast_from(steps);
    let fraction = steps - Vector::<f32, N>::cast_from(truncated);
    let above_half = fraction.greater_than(&Vector::new(0.5f32));
    let tie_to_odd = fraction
        .equal(&Vector::new(0.5f32))
        .vec_and((truncated & Vector::new(1u32)).equal(&Vector::new(1u32)));
    let round_up = Vector::<u32, N>::cast_from(above_half.or(tie_to_odd));
    let subnormal = truncated + round_up;

    let exponent = (magnitude_bits >> Vector::new(F32_MANTISSA_BITS)) + Vector::new(rebias);
    let mantissa = magnitude_bits & Vector::new(F32_MANTISSA_MASK);
    let lsb = (mantissa >> Vector::new(mantissa_shift)) & Vector::new(1u32);
    let rounded =
        ((exponent << Vector::new(F32_MANTISSA_BITS)) | mantissa) + Vector::new(half_ulp - 1) + lsb;
    let normal = rounded >> Vector::new(mantissa_shift);

    let code = select_many(
        magnitude.less_than(&Vector::new(min_normal)),
        subnormal,
        normal,
    );
    let code = select_many(
        magnitude.greater_than(&Vector::new(max_value)),
        Vector::new(max_code),
        code,
    );
    let code = select_many(
        magnitude_bits.greater_than(&Vector::new(F32_INFINITY_BITS)),
        Vector::new(nan_code),
        code,
    );

    code | sign
}

pub type LowerMinifloatCastPass = MatchRewritePass<LowerMinifloatCast>;

#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerMinifloatCast {
    pub native: EnumSet<Fp8Format>,
}

impl LowerMinifloatCast {
    fn emulated(&self, ctx: &Context, value: impl Typed) -> Option<Fp8Format> {
        Fp8Format::of_type(ctx, value.scalar_ty(ctx))
            .filter(|format| !self.native.contains(*format))
    }
}

impl MatchRewrite for LowerMinifloatCast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<CastOp>(ctx) {
            return false;
        }
        let input = op.operand(ctx, 0);
        let result = op.result(ctx);
        // Bool casts belong to the backends' own lowering.
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

fn decode(scope: &Scope, value: Value, format: Fp8Format) -> Value {
    let bytes = reinterpret_value(scope, value, Vector::<u8, N>::__expand_as_type(scope));
    let bits = cast_value(scope, bytes, Vector::<u32, N>::__expand_as_type(scope));
    fp8_bits_to_f32::expand::<N>(scope, bits.into(), format).read_value(scope)
}

fn encode(scope: &Scope, value: Value, format: Fp8Format, result_ty: TypeHandle) -> Value {
    let value = cast_value(scope, value, Vector::<f32, N>::__expand_as_type(scope));
    let bits = f32_to_fp8_bits::expand::<N>(scope, value.into(), format).read_value(scope);
    let bytes = cast_value(scope, bits, Vector::<u8, N>::__expand_as_type(scope));
    reinterpret_value(scope, bytes, result_ty)
}
