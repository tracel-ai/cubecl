use cubecl_ir::{ElemType, Type, Variable};

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
        scope.register_type::<E>(ty.storage_type());
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

#[allow(missing_docs)]
pub fn expand_erf(scope: &Scope, input: Variable, out: Variable) {
    scope.register_type::<ElemA>(input.ty.storage_type());
    scope.register_size::<SizeA>(input.vector_size());
    let res = erf::expand::<ElemA, SizeA>(scope, input.into());
    assign::expand_no_check(scope, res, &mut out.into());
}

#[cube]
fn himul_i64<N: Size>(lhs: Vector<i32, N>, rhs: Vector<i32, N>) -> Vector<i32, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<i64, N>::cast_from(lhs) * Vector::<i64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[cube]
fn himul_u64<N: Size>(lhs: Vector<u32, N>, rhs: Vector<u32, N>) -> Vector<u32, N> {
    let shift = Vector::new(32);
    let mul = (Vector::<u64, N>::cast_from(lhs) * Vector::<u64, N>::cast_from(rhs)) >> shift;
    Vector::cast_from(mul)
}

#[allow(missing_docs)]
pub fn expand_himul_64(scope: &Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_size::<SizeA>(lhs.vector_size());
    match lhs.ty.elem_type() {
        ElemType::Int(_) => {
            let res = himul_i64::expand::<SizeA>(scope, lhs.into(), rhs.into());
            assign::expand_no_check(scope, res, &mut out.into());
        }
        ElemType::UInt(_) => {
            let res = himul_u64::expand::<SizeA>(scope, lhs.into(), rhs.into());
            assign::expand_no_check(scope, res, &mut out.into());
        }
        _ => unreachable!(),
    };
}

#[cube]
fn himul_sim<N: Size>(lhs: Vector<u32, N>, rhs: Vector<u32, N>) -> Vector<u32, N> {
    let low_mask = Vector::new(0xffff);
    let shift = Vector::new(16);

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
pub fn expand_himul_sim(scope: &Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_size::<SizeA>(lhs.vector_size());
    let res = himul_sim::expand::<SizeA>(scope, lhs.into(), rhs.into());
    assign::expand_no_check(scope, res, &mut out.into());
}
