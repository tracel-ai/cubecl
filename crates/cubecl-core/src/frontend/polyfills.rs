use cubecl_ir::{ExpandElement, Variable};

use crate as cubecl;
use crate::prelude::*;

#[cube]
fn checked_index_assign<E: CubePrimitive>(
    index: u32,
    value: Line<E>,
    out: &mut Array<Line<E>>,
    #[comptime] has_buffer_len: bool,
) {
    let array_len = if has_buffer_len {
        out.buffer_len()
    } else {
        out.len()
    };

    if index < array_len {
        out[index] = value;
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
    select(x < Line::new(F::new(0.0)), -erf, erf)
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
    assign::expand(scope, res, ExpandElement::Plain(out).into());
}
