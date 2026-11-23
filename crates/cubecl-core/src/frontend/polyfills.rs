use alloc::vec;
use core::f64::consts::PI;

use cubecl_ir::{Type, cube_op, interfaces::TypedExt, prelude::*};
use num_traits::One;

use crate::prelude::*;
use crate::{self as cubecl, unexpanded};

define_scalar!(ElemA);
define_size!(SizeA);

/// Change the meaning of the given cube primitive type during compilation.
///
/// # Warning
///
/// To be used for very custom kernels, it would likely lead to a JIT compiler error otherwise.
pub fn set_polyfill<E: Scalar, N: Size>(_elem: Type) {
    unexpanded!()
}

/// Expand module of [`set_polyfill()`].
pub mod set_polyfill {
    use super::*;

    /// Expand function of [`set_polyfill()`].
    pub fn expand<E: Scalar, N: Size>(scope: &Scope, ty: Type) {
        scope.register_type::<E>(ty.elem_type());
        scope.register_size::<N>(ty.vector_size());
    }
}

/// Floor modulo using integer arithmetic, preserving precision for supported inputs.
/// Division by zero and signed `MIN % -1` have no portable result or error guarantee.
pub fn expand_signed_mod_floor(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    define_scalar!(I);
    define_size!(N);
    scope.register_value_type::<I, N>(lhs);
    signed_mod_floor::expand::<I, N>(scope, lhs.into(), rhs.into()).read_value(scope)
}

#[cube]
fn signed_mod_floor<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let remainder = lhs % rhs;
    let zero = Vector::new(I::from_int(0));
    let different_signs = (remainder ^ rhs).less_than(&zero);
    let adjust = remainder.not_equal(&zero).vec_and(different_signs);
    remainder + select_many(adjust, rhs, zero)
}

/// Lower NaN-preserving clamp to the corresponding binary operations.
pub fn expand_clamp_nan(scope: &Scope, input: Value, min: Value, max: Value) -> Value {
    use cubecl_ir::dialect::cmp::{FMaxNanOp, FMinNanOp};
    let upper = FMinNanOp::new(scope.ctx_mut(), input, max);
    let upper = scope.register_with_result(&upper);
    let lower = FMaxNanOp::new(scope.ctx_mut(), upper, min);
    scope.register_with_result(&lower)
}

/// Portable lowering for NaN-preserving minimum.
pub fn expand_min_nan(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    define_scalar!(F);
    define_size!(N);
    scope.register_value_type::<F, N>(lhs);
    min_nan_fallback::expand::<F, N>(scope, lhs.into(), rhs.into()).read_value(scope)
}

#[cube]
fn min_nan_fallback<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    let value = lhs.min(rhs);
    let value = select_many(rhs.is_nan(), rhs, value);
    select_many(lhs.is_nan(), lhs, value)
}

/// Portable lowering for NaN-preserving maximum.
pub fn expand_max_nan(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    define_scalar!(F);
    define_size!(N);
    scope.register_value_type::<F, N>(lhs);
    max_nan_fallback::expand::<F, N>(scope, lhs.into(), rhs.into()).read_value(scope)
}

#[cube]
fn max_nan_fallback<F: Float, N: Size>(lhs: Vector<F, N>, rhs: Vector<F, N>) -> Vector<F, N> {
    let value = lhs.max(rhs);
    let value = select_many(rhs.is_nan(), rhs, value);
    select_many(lhs.is_nan(), lhs, value)
}

/// The error function, as FreeBSD's and musl's `erff` compute it (Sun's `s_erff.c`): a rational
/// approximation of `erf` below `0.84375`, of `erf - 0.845` up to `1.25`, and of
/// `log(x · erfc(x)) + x²` up to 6, past which `erf` rounds to ±1. Within a few ulp of the exact
/// value in f32, including the small arguments where `erf(x) ≈ 2x/√π`, and odd: `erf(-0) = -0`.
///
/// Every lane takes all three ranges and selects its own. A NaN fails the comparisons that pick
/// the larger ranges, so it takes the first one, `x + x · P/Q`, and stays NaN. Half floats are
/// evaluated in f32 and rounded once.
#[cube]
pub fn erf<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    if comptime!(F::size().comptime() < 4) {
        Vector::cast_from(erf_ranges::<f32, N>(Vector::cast_from(x)))
    } else {
        erf_ranges::<F, N>(x)
    }
}

#[cube]
fn erf_ranges<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let one = k::<F, N>(1.0);
    let ax = x.abs();

    // |x| < 0.84375: erf(x) = x + x · P(x²)/Q(x²).
    let z = x * x;
    let p = ((((k::<F, N>(-2.3763017452e-05)) * z + k::<F, N>(-5.7702702470e-03)) * z
        + k::<F, N>(-2.8481749818e-02))
        * z
        + k::<F, N>(-3.2504209876e-01))
        * z
        + k::<F, N>(1.2837916613e-01);
    let q = one
        + z * (((((k::<F, N>(-3.9602282413e-06)) * z + k::<F, N>(1.3249473704e-04)) * z
            + k::<F, N>(5.0813062117e-03))
            * z
            + k::<F, N>(6.5022252500e-02))
            * z
            + k::<F, N>(3.9791721106e-01));
    let small = x + x * (p / q);

    // 0.84375 <= |x| < 1.25: erf(|x|) = 0.845 + P(s)/Q(s), with s = |x| - 1.
    let s = ax - one;
    let p = ((((((k::<F, N>(-2.1663755178e-03)) * s + k::<F, N>(3.5478305072e-02)) * s
        + k::<F, N>(-1.1089469492e-01))
        * s
        + k::<F, N>(3.1834661961e-01))
        * s
        + k::<F, N>(-3.7220788002e-01))
        * s
        + k::<F, N>(4.1485610604e-01))
        * s
        + k::<F, N>(-2.3621185683e-03);
    let q = one
        + s * ((((((k::<F, N>(1.1984500103e-02)) * s + k::<F, N>(1.3637083583e-02)) * s
            + k::<F, N>(1.2617121637e-01))
            * s
            + k::<F, N>(7.1828655899e-02))
            * s
            + k::<F, N>(5.4039794207e-01))
            * s
            + k::<F, N>(1.0642088205e-01));
    let mid = k::<F, N>(8.4506291151e-01) + p / q;

    // 1.25 <= |x| < 6: erfc(|x|) = exp(-x² - 0.5625 + R(s)/S(s)) / |x|, with s = 1 / x², and the
    // coefficients for below and above 1/0.35 picked lane by lane.
    let s = one / z;
    let near = ax.less_than(&k::<F, N>(2.857142857));
    let r = (((((((pick::<F, N>(near, -9.8143291473e+00, 0.0)) * s
        + pick::<F, N>(near, -8.1287437439e+01, -4.8351919556e+02))
        * s
        + pick::<F, N>(near, -1.8460508728e+02, -1.0250950928e+03))
        * s
        + pick::<F, N>(near, -1.6239666748e+02, -6.3756646729e+02))
        * s
        + pick::<F, N>(near, -6.2375331879e+01, -1.6063638306e+02))
        * s
        + pick::<F, N>(near, -1.0558626175e+01, -1.7757955551e+01))
        * s
        + pick::<F, N>(near, -6.9385856390e-01, -7.9928326607e-01))
        * s
        + pick::<F, N>(near, -9.8649440333e-03, -9.8649431020e-03);
    let big_s = one
        + s * ((((((((pick::<F, N>(near, -6.0424413532e-02, 0.0)) * s
            + pick::<F, N>(near, 6.5702495575e+00, -2.2440952301e+01))
            * s
            + pick::<F, N>(near, 1.0863500214e+02, 4.7452853394e+02))
            * s
            + pick::<F, N>(near, 4.2900814819e+02, 2.5530502930e+03))
            * s
            + pick::<F, N>(near, 6.4538726807e+02, 3.1998581543e+03))
            * s
            + pick::<F, N>(near, 4.3456588745e+02, 1.5367296143e+03))
            * s
            + pick::<F, N>(near, 1.3765776062e+02, 3.2579251099e+02))
            * s
            + pick::<F, N>(near, 1.9651271820e+01, 3.0338060379e+01));
    let erfc = (-z - k::<F, N>(0.5625) + r / big_s).exp() / ax;
    let tail = one - erfc;

    let above = select_many(
        ax.greater_equal(&k::<F, N>(6.0)),
        one,
        select_many(ax.greater_equal(&k::<F, N>(1.25)), tail, mid),
    );
    let signed = select_many(x.less_than(&k::<F, N>(0.0)), -above, above);
    select_many(ax.greater_equal(&k::<F, N>(0.84375)), signed, small)
}

/// `value` in every lane.
#[cube]
fn k<F: Float, N: Size>(#[comptime] value: f64) -> Vector<F, N> {
    Vector::new(F::new(value))
}

/// `near` in the lanes `mask` sets, `far` in the others.
#[cube]
fn pick<F: Float, N: Size>(
    mask: Vector<bool, N>,
    #[comptime] near: f64,
    #[comptime] far: f64,
) -> Vector<F, N> {
    select_many(mask, k::<F, N>(near), k::<F, N>(far))
}

#[cube]
fn himul_i64<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<i64, N>::cast_from(lhs) * Vector::<i64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[cube]
fn himul_u64<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<u64, N>::cast_from(lhs) * Vector::<u64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[allow(missing_docs)]
pub fn expand_s_himul_64(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    himul_i64::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[allow(missing_docs)]
pub fn expand_u_himul_64(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    himul_u64::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[cube]
fn himul_sim<T: Int, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
    let half_bits = T::size_bits().comptime() / 2;
    let low_mask = Vector::new(T::new((1i64 << half_bits) - 1));
    let shift = Vector::new(T::new(half_bits as i64));

    let lhs_low = lhs & low_mask;
    let lhs_hi = (lhs >> shift) & low_mask;
    let rhs_low = rhs & low_mask;
    let rhs_hi = (rhs >> shift) & low_mask;

    let low_low = lhs_low * rhs_low;
    let high_low = lhs_hi * rhs_low;
    let low_high = lhs_low * rhs_hi;
    let high_high = lhs_hi * rhs_hi;

    let mid = ((low_low >> shift) & low_mask) + (high_low & low_mask) + (low_high & low_mask);
    high_high
        + ((high_low >> shift) & low_mask)
        + ((low_high >> shift) & low_mask)
        + ((mid >> shift) & low_mask)
}

#[allow(missing_docs)]
pub fn expand_himul_sim(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    himul_sim::expand::<ElemA, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

/// Portable implementation of a four-byte signed dot product with accumulation.
#[cube]
pub fn dp4a_polyfill<N: Size>(
    a: Vector<i32, N>,
    b: Vector<i32, N>,
    c: Vector<i32, N>,
) -> Vector<i32, N> {
    let a = Vector::<u32, N>::reinterpret(a);
    let b = Vector::<u32, N>::reinterpret(b);
    let shift_8 = Vector::new(8);
    let shift_16 = Vector::new(16);
    let shift_24 = Vector::new(24);
    let byte_mask = Vector::new(0xff);
    let sign_mask = Vector::new(0x80);
    let sign_offset = Vector::<i32, N>::new(0x80);

    let a0 = Vector::<i32, N>::cast_from((a & byte_mask) ^ sign_mask) - sign_offset;
    let a1 = Vector::<i32, N>::cast_from(((a >> shift_8) & byte_mask) ^ sign_mask) - sign_offset;
    let a2 = Vector::<i32, N>::cast_from(((a >> shift_16) & byte_mask) ^ sign_mask) - sign_offset;
    let a3 = Vector::<i32, N>::cast_from((a >> shift_24) ^ sign_mask) - sign_offset;

    let b0 = Vector::<i32, N>::cast_from((b & byte_mask) ^ sign_mask) - sign_offset;
    let b1 = Vector::<i32, N>::cast_from(((b >> shift_8) & byte_mask) ^ sign_mask) - sign_offset;
    let b2 = Vector::<i32, N>::cast_from(((b >> shift_16) & byte_mask) ^ sign_mask) - sign_offset;
    let b3 = Vector::<i32, N>::cast_from((b >> shift_24) ^ sign_mask) - sign_offset;

    c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3
}

#[allow(missing_docs)]
pub fn expand_dp4a_polyfill(scope: &Scope, a: Value, b: Value, c: Value) -> Value {
    scope.register_size::<SizeA>(a.vector_size(scope.ctx()));
    dp4a_polyfill::expand::<SizeA>(scope, a.into(), b.into(), c.into()).value(scope)
}

#[cube]
pub fn log1p<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    (input + Vector::one()).ln()
}

#[cube]
pub fn expm1<T: Float, N: Size>(x: Vector<T, N>) -> Vector<T, N> {
    let sq = x * x;
    let a = sq * Vector::new(T::new(0.5));
    let b = sq * x * Vector::new(T::new(1.0 / 6.0));
    let taylor = x + a + b;
    let is_small = x.abs().less_than(&Vector::new(T::new(1e-5)));
    select_many(is_small, taylor, x.exp() - Vector::one())
}

/// `powf` without any edge case handling. Useful as a common mapping for the backend version that
/// doesn't handle edge cases normally.
#[cube_op(name = "polyfill.simple_pow")]
#[result_ty(same_as = base)]
pub struct SimplePowOp {
    pub base: Value,
    pub exp: Value,
}

/// use the simple version because otherwise we'd get an infinite lowering loop
#[cube]
fn simple_pow<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<T, N>) -> Vector<T, N> {
    intrinsic!(|scope| {
        let base = base.read_value(scope);
        let exp = exp.read_value(scope);
        let powf = SimplePowOp::new(scope.ctx_mut(), base, exp);
        scope.register_with_result(&powf).into()
    })
}

#[cube]
pub fn powf<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<T, N>) -> Vector<T, N> {
    let modulo = exp.mod_floor(Vector::new(T::new(2.0)));
    let is_even = modulo.equal(&Vector::zero());
    let is_odd = modulo.equal(&Vector::one());
    let is_neg_base = base.less_than(&Vector::zero());

    let even_res = simple_pow(base.abs(), exp);
    let odd_neg_res = -(simple_pow(-base, exp));
    let default = simple_pow(base, exp);

    let sel1 = select_many(is_odd.vec_and(is_neg_base), odd_neg_res, default);
    select_many(is_even, even_res, sel1)
}

/// Integer powers by squaring, without converting the exponent to a float.
/// Constant exponents, including vector splats, expand to multiplication chains.
#[cube]
pub fn powi<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    intrinsic!(|scope| {
        let value = exp.read_value(scope);
        if let Some(exponent) = powi_constant_exponent(scope.ctx(), value) {
            powi_constant::expand::<T, N>(scope, base, exponent)
        } else {
            powi_runtime::expand::<T, N>(scope, base, exp)
        }
    })
}

// Vector literals are broadcasts of scalar constants at this point in lowering.
fn powi_constant_exponent(ctx: &Context, mut value: Value) -> Option<i32> {
    use cubecl_ir::{attributes::IntAttrExt, dialect::vector::VectorBroadcastOp};
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};

    loop {
        let op = value.defining_op()?;
        if let Some(broadcast) = op.as_op::<VectorBroadcastOp>(ctx) {
            value = broadcast.input(ctx);
        } else {
            let attr = op
                .as_op::<ConstantOp>(ctx)?
                .get_attr_builtin_constant_value(ctx)?;
            return attr.downcast_ref::<IntegerAttr>()?.as_value::<i32>(ctx);
        }
    }
}

// Emit the squaring chain for a known exponent. In particular, a square
// is one multiply for both scalar constants and vector splats, with no loop.
#[cube]
fn powi_constant<T: Float, N: Size>(base: Vector<T, N>, #[comptime] exp: i32) -> Vector<T, N> {
    // For negative exponents, form base^(|exp| / 2), then square its reciprocal.
    // This postpones reciprocal rounding without overflowing the full positive
    // power when the negative power is a representable subnormal.
    let magnitude = comptime!(exp.unsigned_abs() >> u32::from(exp < 0));
    let mut factor = base;
    let mut result = Vector::one();
    #[unroll]
    for bit in 0..(32 - magnitude.leading_zeros()) {
        if magnitude & (1 << bit) != 0 {
            if bit == magnitude.trailing_zeros() {
                result = factor;
            } else {
                result *= factor;
            }
        }
        if bit + 1 < 32 - magnitude.leading_zeros() {
            factor *= factor;
        }
    }
    if exp < 0 {
        let reciprocal = Vector::one() / result;
        result = reciprocal * reciprocal;
        if exp & 1 != 0 {
            result /= base;
        }
    }
    result
}

#[cube]
fn powi_runtime<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    let zero = Vector::<u32, N>::zero();
    let one = Vector::<T, N>::one();
    let one_u = Vector::<u32, N>::one();
    let negative = exp.less_than(&Vector::zero());

    // Unsigned subtraction also represents abs(i32::MIN), without signed overflow.
    let unsigned = Vector::<u32, N>::cast_from(exp);
    let magnitude = select_many(negative, zero - unsigned, unsigned);
    // As in the constant path, negative powers use half the magnitude so the
    // reciprocal can be taken after exponentiation without losing subnormals.
    let mut remaining = select_many(negative, magnitude >> one_u, magnitude);
    let mut factor = base;
    // Consume the first bit without multiplying by one. Square only when another
    // exponent bit remains, so a runtime square needs just one loop iteration.
    let mut result = select_many((remaining & one_u).equal(&one_u), base, one);
    remaining >>= one_u;

    // The highest bit across all lanes bounds the loop, without a reduction on
    // every iteration. OR the signed exponents too: a negative result means at
    // least one lane needs a reciprocal.
    let mut bits = 0u32;
    let mut signs = 0i32;
    #[unroll]
    for lane in 0..remaining.vector_size() {
        bits |= remaining.extract(lane);
        signs |= exp.extract(lane);
    }
    while bits != 0 {
        factor *= factor;
        // Select after multiplying so finished lanes retain their result even
        // if their unused factor overflows while other lanes continue.
        result = select_many((remaining & one_u).equal(&one_u), result * factor, result);
        remaining >>= one_u;
        bits >>= 1;
    }

    if signs < 0 {
        let reciprocal = one / result;
        let divisor = select_many((magnitude & one_u).equal(&one_u), base, one);
        result = select_many(negative, (reciprocal * reciprocal) / divisor, result);
    }
    result
}

/// Integer power with a signed exponent.
///
/// A non-negative exponent gives the wrapping power, like [`powi_unsigned`]. A negative exponent
/// gives `1 / base^|exp|` truncated toward zero: 1 for a base of 1, 1 or -1 by the exponent's
/// parity for a base of -1, and 0 for any other base, including 0.
#[cube]
pub fn powi_int<T: Int, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    let one = Vector::<T, N>::one();
    // Read as `u32`, a negative exponent keeps its parity, so the wrapping power is already the
    // truncated one for a base of 1 or -1.
    let power = powi_unsigned(base, Vector::<u32, N>::cast_from(exp));

    // Only a signed base can be -1. For an unsigned one, `T::new(-1)` would be its maximum value.
    let dtype = elem_type_of::<T>().comptime();
    let unit_base = if dtype.is_signed_int() {
        base.equal(&one).or(base.equal(&Vector::new(T::new(-1))))
    } else {
        base.equal(&one)
    };
    let truncated = select_many(unit_base, power, Vector::zero());

    select_many(exp.less_than(&Vector::zero()), truncated, power)
}

/// Wrapping integer power with an unsigned exponent, like the primitive `wrapping_pow`.
#[cube]
pub fn powi_unsigned<T: Int, N: Size>(base: Vector<T, N>, exp: Vector<u32, N>) -> Vector<T, N> {
    // Use unsigned arithmetic so intermediate squares wrap on C++ backends too.
    // Narrow integers can use u32: truncating the final result preserves their low bits
    // and avoids C++ integer promotion turning multiplication into signed arithmetic.
    if T::size_bits().comptime() <= 32 {
        Vector::cast_from(powi_unsigned_wide::<u32, N>(Vector::cast_from(base), exp))
    } else {
        Vector::cast_from(powi_unsigned_wide::<u64, N>(Vector::cast_from(base), exp))
    }
}

// Only instantiate with u32 or u64 to keep multiplication unsigned on every backend.
#[cube]
fn powi_unsigned_wide<T: Int, N: Size>(base: Vector<T, N>, exp: Vector<u32, N>) -> Vector<T, N> {
    let one_u = Vector::<u32, N>::new(1);
    let one_t = Vector::<T, N>::new(T::from_int(1));

    let mut exp = exp;
    let mut result = one_t;
    let mut factor = base;

    #[unroll]
    for _ in 0..32 {
        // TODO: implement peephole optimization for masked multiplication
        result *= select_many((exp & one_u).equal(&one_u), factor, one_t);
        factor *= factor;
        exp >>= one_u;
    }

    result
}

#[cube]
pub fn recip<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    Vector::one() / input
}

/// Cube root of the magnitude with the input's sign put back, since `powf` is NaN for a negative
/// base. A backend `pow` can be a few ulp off (WGSL's is `exp2(y * log2(x))`), so one Newton step
/// refines it. The step is written as `mag / y²` rather than `y³` so it can't overflow, and is
/// skipped for zero and infinity, where it would divide by zero.
#[cube]
pub fn cbrt<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    let zero = Vector::<T, N>::zero();
    let mag = input.abs();
    let approx = mag.powf(Vector::new(T::new(1.0 / 3.0)));
    let refined = (approx + approx + mag / (approx * approx)) / Vector::new(T::new(3.0));
    let root = select_many(mag.equal(&zero).or(mag.is_inf()), approx, refined);
    select_many(input.less_than(&zero), -root, root)
}

#[cube]
pub fn inverse_cbrt<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    Vector::one() / cbrt(input)
}

#[cube]
pub fn to_degrees<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    input * Vector::new(T::new(comptime!(180.0 / PI)))
}

#[cube]
pub fn to_radians<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    input * Vector::new(T::new(comptime!(PI / 180.0)))
}

pub mod bitwise {
    use super::*;

    #[cube]
    pub fn u64_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
        let shift = Vector::new(I::new(32));

        let low = Vector::<u32, N>::cast_from(x);
        let high = Vector::<u32, N>::cast_from(x >> shift);
        let low_zeros = Vector::leading_zeros(low);
        let high_zeros = Vector::leading_zeros(high);

        select_many(
            high_zeros.equal(&Vector::new(32)),
            low_zeros + high_zeros,
            high_zeros,
        )
    }

    #[cube]
    pub fn u64_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
        let shift = Vector::new(I::new(32));

        let low = Vector::<u32, N>::cast_from(x);
        let high = Vector::<u32, N>::cast_from(x >> shift);
        let low_tz = Vector::trailing_zeros(low);
        let high_tz = Vector::trailing_zeros(high);

        let high_tz = select_many(
            high_tz.equal(&Vector::new(32)),
            Vector::new(64),
            high_tz + Vector::new(32),
        );
        select_many(low_tz.equal(&Vector::new(32)), high_tz, low_tz)
    }

    #[cube]
    pub fn u64_ffs<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
        let shift = Vector::new(I::new(32));

        let low = Vector::<u32, N>::cast_from(x);
        let high = Vector::<u32, N>::cast_from(x >> shift);
        let low_ffs = Vector::find_first_set(low);
        let high_ffs = Vector::find_first_set(high);

        let high_ffs = select_many(
            high_ffs.equal(&Vector::new(0)),
            high_ffs,
            high_ffs + Vector::new(32),
        );
        select_many(low_ffs.equal(&Vector::new(0)), high_ffs, low_ffs)
    }
}

/// The plane reductions and scans, as folds over the shuffles.
///
/// A backend that has cross-lane shuffles but no reduction of its own gets them from here; the
/// C++ backends and the LLVM one share these.
pub mod plane {
    use super::*;
    use crate::prelude::{
        CUBE_DIM, CubeAdd, CubeMul, CubePartialOrd, PLANE_DIM, UNIT_POS_PLANE, max, min,
        plane_shuffle_up, plane_shuffle_xor, select,
    };

    #[cube]
    pub trait PlaneOp<T: Scalar, N: Size> {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N>;
    }

    pub struct OpAdd;
    pub struct OpMul;
    pub struct OpMin;
    pub struct OpMax;

    #[cube]
    impl<T: Scalar + CubeAdd, N: Size> PlaneOp<T, N> for OpAdd {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            lhs + rhs
        }
    }
    #[cube]
    impl<T: Scalar + CubeMul, N: Size> PlaneOp<T, N> for OpMul {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            lhs * rhs
        }
    }
    #[cube]
    impl<T: Scalar + CubePartialOrd, N: Size> PlaneOp<T, N> for OpMin {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            min(lhs, rhs)
        }
    }
    #[cube]
    impl<T: Scalar + CubePartialOrd, N: Size> PlaneOp<T, N> for OpMax {
        fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
            max(lhs, rhs)
        }
    }

    #[cube]
    fn plane_dim_checked() -> u32 {
        min(PLANE_DIM, CUBE_DIM)
    }

    #[cube]
    pub fn plane_reduce<T: Scalar, N: Size, Op: PlaneOp<T, N>>(val: Vector<T, N>) -> Vector<T, N> {
        let plane_dim = plane_dim_checked();
        let mut acc = val;
        let mut offset = 1;
        while offset < plane_dim {
            acc = Op::apply(acc, plane_shuffle_xor(acc, offset));
            offset *= 2;
        }
        acc
    }

    #[cube]
    pub fn plane_reduce_inclusive<T: Scalar, N: Size, Op: PlaneOp<T, N>>(
        val: Vector<T, N>,
    ) -> Vector<T, N> {
        let plane_dim = plane_dim_checked();
        let mut acc = val;
        let mut offset = 1;
        while offset < plane_dim {
            let tmp = Op::apply(acc, plane_shuffle_up(acc, offset));
            if UNIT_POS_PLANE >= offset {
                acc = tmp;
            }
            offset *= 2;
        }
        acc
    }

    #[cube]
    pub fn plane_reduce_exclusive<T: Numeric, N: Size, Op: PlaneOp<T, N>>(
        val: Vector<T, N>,
        #[comptime] default: i64,
    ) -> Vector<T, N> {
        let inclusive = plane_reduce_inclusive::<T, N, Op>(val);
        let shfl = plane_shuffle_up(inclusive, 1);
        select(UNIT_POS_PLANE == 0, Vector::new(T::from_int(default)), shfl)
    }
}

/// Where a lane's registers sit in the tile, for the manual `mma.sync` matrix API.
///
/// Unlike the cooperative API, whose fragment layout is the hardware's business, the manual one
/// hands a kernel the registers and expects it to know which element of the tile each holds.
/// NVIDIA documents that mapping, and every backend generating `mma.sync` has to agree with it
/// exactly -- a kernel that indexes its own fragment differently from the way the instruction
/// reads it computes a wrong answer rather than failing -- so it is written once here and the
/// C++ and LLVM backends both expand it.
///
/// Derived from the PTX shape documentation:
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma>
pub mod mma {
    use super::*;
    use crate::ir::types::MatrixIdent;

    /// The row of the tile that `lane_id`'s `i`th element holds.
    #[cube]
    pub fn row_index(
        lane_id: u32,
        i: u32,
        #[comptime] elems_per_reg: usize,
        #[comptime] ident: MatrixIdent,
    ) -> u32 {
        let elems_per_reg = elems_per_reg as u32;
        match ident {
            MatrixIdent::A => {
                let group_id = lane_id / 4;
                let odd_register = (i / elems_per_reg) & 1;
                group_id + odd_register * 8
            }
            MatrixIdent::B => {
                let thread_id_in_group = lane_id % 4;
                let offset = thread_id_in_group * elems_per_reg + (i % elems_per_reg);
                let reg = i / elems_per_reg;
                offset + elems_per_reg * 4 * reg
            }
            MatrixIdent::Accumulator => {
                let group_id = lane_id / 4;
                let offset = (i << 2) & 8;
                group_id + offset
            }
        }
    }

    /// The column of the tile that `lane_id`'s `i`th element holds.
    #[cube]
    pub fn col_index(
        lane_id: u32,
        i: u32,
        #[comptime] elems_per_reg: usize,
        #[comptime] ident: MatrixIdent,
    ) -> u32 {
        let elems_per_reg = elems_per_reg as u32;
        match ident {
            MatrixIdent::A => {
                let thread_id_in_group = lane_id % 4;
                let offset = thread_id_in_group * elems_per_reg + (i % elems_per_reg);
                let group_2 = (i / (2 * elems_per_reg)) & 1;
                offset + 4 * elems_per_reg * group_2
            }
            MatrixIdent::B => lane_id >> 2,
            MatrixIdent::Accumulator => {
                let thread_id_in_group = lane_id % 4;
                (thread_id_in_group * 2) + (i % 2)
            }
        }
    }
}
