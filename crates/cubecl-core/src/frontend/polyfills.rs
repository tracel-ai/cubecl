use alloc::vec;

use cubecl_ir::{Type, cube_op, interfaces::TypedExt, prelude::*};
use num_traits::One;

use crate::prelude::*;
use crate::{self as cubecl, unexpanded};

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

#[cube]
pub fn erf<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let erf = erf_positive(x.abs());
    select_many(x.less_than(&Vector::new(F::new(0f32))), -erf, erf)
}

/// An approximation of the error function: <https://en.wikipedia.org/wiki/Error_function#Numerical_approximations>
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
#[cube]
fn erf_positive<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let p = Vector::new(F::new(0.3275911_f32));
    let a1 = Vector::new(F::new(0.2548296_f32));
    let a2 = Vector::new(F::new(-0.28449674_f32));
    let a3 = Vector::new(F::new(1.4214137_f32));
    let a4 = Vector::new(F::new(-1.453152_f32));
    let a5 = Vector::new(F::new(1.0614054_f32));
    let one = Vector::new(F::new(1.0_f32));

    let t = one / (one + p * x);
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    one - (tmp * t * (-x * x).exp())
}

#[cube]
fn himul_i64<N: Size>(lhs: Vector<i32, N>, rhs: Vector<i32, N>) -> Vector<i32, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<i64, N>::cast_from(lhs) * Vector::<i64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[cube]
pub fn himul_u64<N: Size>(lhs: Vector<u32, N>, rhs: Vector<u32, N>) -> Vector<u32, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<u64, N>::cast_from(lhs) * Vector::<u64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[allow(missing_docs)]
pub fn expand_s_himul_64(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_size::<SizeA>(lhs.vector_size(scope.ctx()));
    himul_i64::expand::<SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[allow(missing_docs)]
pub fn expand_u_himul_64(scope: &Scope, lhs: Value, rhs: Value) -> Value {
    scope.register_size::<SizeA>(lhs.vector_size(scope.ctx()));
    himul_u64::expand::<SizeA>(scope, lhs.into(), rhs.into()).value(scope)
}

#[cube]
fn himul_sim<T: Int, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
    let half_bits = comptime!(T::size_bits() / 2);
    let low_mask = Vector::new(T::new(comptime!((1 << half_bits) - 1)));
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
    scope.register_size::<SizeA>(lhs.vector_size(scope.ctx()));
    if lhs.is_int_of_width(scope.ctx(), 32) {
        himul_sim::expand::<u32, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
    } else {
        himul_sim::expand::<u64, SizeA>(scope, lhs.into(), rhs.into()).value(scope)
    }
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

#[cube]
pub fn powi<T: Float, N: Size>(base: Vector<T, N>, exp: Vector<i32, N>) -> Vector<T, N> {
    let is_even = exp.is_multiple_of(2);
    let is_neg_base = base.less_than(&Vector::zero());
    let exp = Vector::cast_from(exp);

    let even_res = simple_pow(base.abs(), exp);
    let odd_neg_res = -(simple_pow(-base, exp));
    let default = simple_pow(base, exp);

    let sel1 = select_many((!is_even).vec_and(is_neg_base), odd_neg_res, default);
    select_many(is_even, even_res, sel1)
}

#[cube]
pub fn recip<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    Vector::one() / input
}
