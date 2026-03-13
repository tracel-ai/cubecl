use cubecl_ir::{ElemType, ManagedVariable, Type, Variable};

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
    pub fn expand<E: Scalar, N: Size>(scope: &mut Scope, ty: Type) {
        scope.register_type::<E>(ty.storage_type());
        scope.register_size::<N>(ty.vector_size());
    }
}

#[cube]
fn checked_index_assign<E: Scalar, N: Size>(
    index: usize,
    value: Vector<E, N>,
    out: &mut Array<Vector<E, N>>,
    #[comptime] has_buffer_len: bool,
    #[comptime] unroll_factor: usize,
) {
    let array_len = if has_buffer_len {
        out.buffer_len()
    } else {
        out.len()
    };

    if index < array_len * unroll_factor {
        unsafe { out.index_assign_unchecked(index, value) };
    }
}

#[allow(missing_docs)]
pub fn expand_checked_index_assign(
    scope: &mut Scope,
    lhs: Variable,
    rhs: Variable,
    out: Variable,
    unroll_factor: usize,
) {
    scope.register_type::<ElemA>(rhs.ty.storage_type());
    scope.register_size::<SizeA>(rhs.ty.vector_size());
    checked_index_assign::expand::<ElemA, SizeA>(
        scope,
        ManagedVariable::Plain(lhs).into(),
        ManagedVariable::Plain(rhs).into(),
        ManagedVariable::Plain(out).into(),
        out.has_buffer_length(),
        unroll_factor,
    );
}

#[cube]
pub fn erf<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let erf = erf_positive(x.abs());
    select_many(x.less_than(Vector::new(F::new(0.0))), -erf, erf)
}

/// An approximation of the error function: <https://en.wikipedia.org/wiki/Error_function#Numerical_approximations>
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
#[cube]
fn erf_positive<F: Float, N: Size>(x: Vector<F, N>) -> Vector<F, N> {
    let p = Vector::new(F::new(0.3275911));
    let a1 = Vector::new(F::new(0.2548296));
    let a2 = Vector::new(F::new(-0.28449674));
    let a3 = Vector::new(F::new(1.4214137));
    let a4 = Vector::new(F::new(-1.453152));
    let a5 = Vector::new(F::new(1.0614054));
    let one = Vector::new(F::new(1.0));

    let t = one / (one + p * x);
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    one - (tmp * t * (-x * x).exp())
}

#[allow(missing_docs)]
pub fn expand_erf(scope: &mut Scope, input: Variable, out: Variable) {
    scope.register_type::<ElemA>(input.ty.storage_type());
    scope.register_size::<SizeA>(input.vector_size());
    let res = erf::expand::<ElemA, SizeA>(scope, ManagedVariable::Plain(input).into());
    assign::expand_no_check(scope, res, ManagedVariable::Plain(out).into());
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
pub fn expand_himul_64(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_size::<SizeA>(lhs.vector_size());
    match lhs.ty.elem_type() {
        ElemType::Int(_) => {
            let res = himul_i64::expand::<SizeA>(
                scope,
                ManagedVariable::Plain(lhs).into(),
                ManagedVariable::Plain(rhs).into(),
            );
            assign::expand_no_check(scope, res, ManagedVariable::Plain(out).into());
        }
        ElemType::UInt(_) => {
            let res = himul_u64::expand::<SizeA>(
                scope,
                ManagedVariable::Plain(lhs).into(),
                ManagedVariable::Plain(rhs).into(),
            );
            assign::expand_no_check(scope, res, ManagedVariable::Plain(out).into());
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
pub fn expand_himul_sim(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_size::<SizeA>(lhs.vector_size());
    let res = himul_sim::expand::<SizeA>(
        scope,
        ManagedVariable::Plain(lhs).into(),
        ManagedVariable::Plain(rhs).into(),
    );
    assign::expand_no_check(scope, res, ManagedVariable::Plain(out).into());
}
