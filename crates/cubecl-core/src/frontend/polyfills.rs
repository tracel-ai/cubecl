use cubecl_ir::{Elem, ExpandElement, Variable};

use crate::prelude::*;
use crate::{self as cubecl, unexpanded};

/// Change the meaning of the given cube primitive type during compilation.
///
/// # Warning
///
/// To be used for very custom kernels, it would likely lead to a JIT compiler error otherwise.
pub fn set_polyfill<E: CubePrimitive>(_elem: Elem) {
    unexpanded!()
}

/// Expand module of [set_polyfill()].
pub mod set_polyfill {
    use super::*;

    /// Expand function of [set_polyfill()].
    pub fn expand<E: CubePrimitive>(scope: &mut Scope, elem: Elem) {
        scope.register_elem::<E>(elem);
    }
}

#[cube]
fn checked_index_assign<E: CubePrimitive>(
    index: u32,
    value: Line<E>,
    out: &mut Array<Line<E>>,
    #[comptime] has_buffer_len: bool,
) {
    let array_len = if comptime![has_buffer_len] {
        out.buffer_len()
    } else {
        out.len()
    };

    if index < array_len {
        unsafe { out.index_assign_unchecked(index, value) };
    }
}

#[allow(missing_docs)]
pub fn expand_checked_index_assign(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    scope.register_elem::<FloatExpand<0>>(rhs.item.elem);
    checked_index_assign::expand::<FloatExpand<0>>(
        scope,
        ExpandElement::Plain(lhs).into(),
        ExpandElement::Plain(rhs).into(),
        ExpandElement::Plain(out).into(),
        out.has_buffer_length(),
    );
}

#[cube]
pub fn erf<F: Float>(x: Line<F>) -> Line<F> {
    let erf = erf_positive(Abs::abs(x));
    select_many(x.less_than(Line::new(F::new(0.0))), -erf, erf)
}

/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
#[cube]
fn erf_positive<F: Float>(x: Line<F>) -> Line<F> {
    let p = Line::new(F::new(0.3275911));
    let a1 = Line::new(F::new(0.2548296));
    let a2 = Line::new(F::new(-0.28449674));
    let a3 = Line::new(F::new(1.4214137));
    let a4 = Line::new(F::new(-1.453152));
    let a5 = Line::new(F::new(1.0614054));
    let one = Line::new(F::new(1.0));

    let t = one / (one + p * x);
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    one - (tmp * t * Exp::exp(-x * x))
}

#[allow(missing_docs)]
pub fn expand_erf(scope: &mut Scope, input: Variable, out: Variable) {
    scope.register_elem::<FloatExpand<0>>(input.item.elem);
    let res = erf::expand::<FloatExpand<0>>(scope, ExpandElement::Plain(input).into());
    assign::expand_no_check(scope, res, ExpandElement::Plain(out).into());
}

#[cube]
fn himul_i64(lhs: Line<i32>, rhs: Line<i32>) -> Line<i32> {
    let shift = Line::empty(lhs.size()).fill(32);
    let mul = (Line::<i64>::cast_from(lhs) * Line::<i64>::cast_from(rhs)) >> shift;
    Line::cast_from(mul)
}

#[cube]
fn himul_u64(lhs: Line<u32>, rhs: Line<u32>) -> Line<u32> {
    let shift = Line::empty(lhs.size()).fill(32);
    let mul = (Line::<u64>::cast_from(lhs) * Line::<u64>::cast_from(rhs)) >> shift;
    Line::cast_from(mul)
}

#[allow(missing_docs)]
pub fn expand_himul_64(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable) {
    match lhs.item.elem {
        Elem::Int(_) => {
            let res = himul_i64::expand(
                scope,
                ExpandElement::Plain(lhs).into(),
                ExpandElement::Plain(rhs).into(),
            );
            assign::expand_no_check(scope, res, ExpandElement::Plain(out).into());
        }
        Elem::UInt(_) => {
            let res = himul_u64::expand(
                scope,
                ExpandElement::Plain(lhs).into(),
                ExpandElement::Plain(rhs).into(),
            );
            assign::expand_no_check(scope, res, ExpandElement::Plain(out).into());
        }
        _ => unreachable!(),
    };
}

#[cube]
fn himul_sim(lhs: Line<u32>, rhs: Line<u32>) -> Line<u32> {
    let low_mask = Line::empty(lhs.size()).fill(0xffff);
    let shift = Line::empty(lhs.size()).fill(16);

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
    let res = himul_sim::expand(
        scope,
        ExpandElement::Plain(lhs).into(),
        ExpandElement::Plain(rhs).into(),
    );
    assign::expand_no_check(scope, res, ExpandElement::Plain(out).into());
}
