//! Conversion to `bf16` from a source wider than `f32`.
//!
//! A backend converts to `bf16` from `f32` with one rounding, but from an `f64` or a 32- or 64-bit
//! integer it goes through the nearest `f32` and rounds twice: a value just past a `bf16` tie can
//! round onto the tie in `f32`, then to even the wrong way. [`LowerWideBf16Cast`] narrows such a
//! source to `f32` by rounding to odd instead, which `f32`'s width keeps exact for the conversion
//! to `bf16` that follows.

use alloc::vec;
use cubecl_ir::{
    NamedRewrite, Scope, dialect::general::CastOp, interfaces::TypedExt, prelude::*,
    types::scalar::BFloat16Type,
};
use pliron::r#type::TypeHandle;

use crate::{self as cubecl, prelude::*};

define_size!(N);

const F32_SIGNIFICAND_BITS: u32 = f32::MANTISSA_DIGITS;
const F32_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
const F32_EXPONENT_BIAS: u32 = (f32::MAX_EXP - 1) as u32;

pub type LowerWideBf16CastPass = MatchRewritePass<LowerWideBf16Cast>;

/// Rewrites a cast to `bf16` from a source wider than `f32` as the cast from an `f32` rounded to
/// odd, which the backend converts natively with the one rounding the source needs.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerWideBf16Cast;

impl MatchRewrite for LowerWideBf16Cast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<CastOp>(ctx)
            && is_bf16(ctx, op.result(ctx))
            && needs_rounding_to_odd(ctx, op.operand(ctx, 0).scalar_ty(ctx))
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
        let lanes = input.vector_size(ctx);
        debug_assert_eq!(
            lanes,
            result_ty.vector_size(ctx),
            "A cast keeps its vectorization, so one `N` describes both sides"
        );
        scope.register_size::<N>(lanes);

        let value = cast_value(&scope, to_f32_rounding_once(&scope, input), result_ty);
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

/// Narrows to `f32` by rounding to odd: an inexact result keeps its low bit set, so rounding it
/// once more to `bf16` lands where rounding `value` straight to `bf16` would. Rounding to the
/// nearest `f32` first would round twice.
#[cube]
fn f64_to_f32_round_to_odd<N: Size>(value: Vector<f64, N>) -> Vector<f32, N> {
    let nearest = Vector::<f32, N>::cast_from(value);
    let widened = Vector::<f64, N>::cast_from(nearest);
    let bits = Vector::<u32, N>::reinterpret(nearest);
    // The bits of an `f32` order as its magnitude does, so a step down is one ulp toward zero.
    let rounded_away = widened.abs().greater_than(&value.abs());
    let toward_zero = select_many(rounded_away, bits - Vector::new(1u32), bits);
    let odd = select_many(widened.equal(&value), bits, toward_zero | Vector::new(1u32));
    Vector::<f32, N>::reinterpret(odd)
}

/// [`f64_to_f32_round_to_odd`] for an unsigned integer: its top 24 significant bits, with a
/// sticky low bit for whatever was dropped below them. Generic over the width, so a 32-bit
/// source does not pay for 64-bit arithmetic.
#[cube]
fn unsigned_to_f32_round_to_odd<U: Int, N: Size>(magnitude: Vector<U, N>) -> Vector<f32, N> {
    let bits = Vector::new(U::size_bits().comptime() as u32);
    let significant = bits - Vector::leading_zeros(magnitude);
    let excess = select_many(
        significant.greater_than(&Vector::new(F32_SIGNIFICAND_BITS)),
        significant - Vector::new(F32_SIGNIFICAND_BITS),
        Vector::new(0u32),
    );
    let shift = Vector::<U, N>::cast_from(excess);
    let kept = magnitude >> shift;
    let dropped = (kept << shift).not_equal(&magnitude);
    let odd = kept | Vector::<U, N>::cast_from(dropped);
    // `odd` fits the significand and the scale is a power of two: neither step rounds.
    let scale = Vector::<f32, N>::reinterpret(
        (excess + Vector::new(F32_EXPONENT_BIAS)) << Vector::new(F32_MANTISSA_BITS),
    );
    Vector::<f32, N>::cast_from(odd) * scale
}

/// [`unsigned_to_f32_round_to_odd`] on the magnitude, the sign put back after. `U` is the
/// unsigned type of `S`'s width.
#[cube]
fn signed_to_f32_round_to_odd<S: Int, U: Int, N: Size>(value: Vector<S, N>) -> Vector<f32, N> {
    let negative = value.less_than(&Vector::zero());
    // The most negative value negates to itself, whose bits are its magnitude.
    let magnitude =
        Vector::<U, N>::reinterpret(select_many(negative, Vector::zero() - value, value));
    let rounded = unsigned_to_f32_round_to_odd::<U, N>(magnitude);
    select_many(negative, Vector::new(0.0f32) - rounded, rounded)
}

fn is_bf16(ctx: &Context, value: impl Typed) -> bool {
    value
        .try_get_scalar_ty(ctx)
        .is_some_and(|scalar| scalar.deref(ctx).is::<BFloat16Type>())
}

/// `N` lanes as an `f32` that rounds to `bf16` as the lanes themselves would. Only a source with
/// more significant bits than `f32` needs rounding to odd; any other converts exactly.
fn to_f32_rounding_once(scope: &Scope, value: Value) -> Value {
    let scalar = value.scalar_ty(scope.ctx());
    if !needs_rounding_to_odd(scope.ctx(), scalar) {
        return cast_value(scope, value, Vector::<f32, N>::__expand_as_type(scope));
    }
    let signed = scalar.is_signed_int(scope.ctx());
    if scalar.is_float64(scope.ctx()) {
        let value = cast_value(scope, value, Vector::<f64, N>::__expand_as_type(scope));
        f64_to_f32_round_to_odd::expand::<N>(scope, value.into()).read_value(scope)
    } else if scalar.size_bits(scope.ctx()) <= u32::BITS as usize {
        int_to_f32_round_to_odd::<i32, u32>(scope, value, signed)
    } else {
        int_to_f32_round_to_odd::<i64, u64>(scope, value, signed)
    }
}

/// [`signed_to_f32_round_to_odd`] or [`unsigned_to_f32_round_to_odd`] at the width of `S`/`U`.
fn int_to_f32_round_to_odd<S: Int, U: Int>(scope: &Scope, value: Value, signed: bool) -> Value {
    match signed {
        true => {
            let value = cast_value(scope, value, Vector::<S, N>::__expand_as_type(scope));
            signed_to_f32_round_to_odd::expand::<S, U, N>(scope, value.into()).read_value(scope)
        }
        false => {
            let value = cast_value(scope, value, Vector::<U, N>::__expand_as_type(scope));
            unsigned_to_f32_round_to_odd::expand::<U, N>(scope, value.into()).read_value(scope)
        }
    }
}

/// Whether a `scalar` source carries more significant bits than `f32`, which would round twice on
/// its way to `bf16` through a nearest `f32`.
fn needs_rounding_to_odd(ctx: &Context, scalar: TypeHandle) -> bool {
    let integer = scalar.is_int(ctx) || scalar.is_index(ctx);
    scalar.is_float64(ctx) || integer && scalar.size_bits(ctx) > F32_SIGNIFICAND_BITS as usize
}
