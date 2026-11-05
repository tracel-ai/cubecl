//! `sinh`, `cosh` and `tanh` to 3.5 ulp, and `asinh`, `acosh` and `atanh` to 1 ulp, from SLEEF
//! `xsinh_u35`, `xcosh_u35`, `xtanh_u35`, `xasinh`, `xacosh` and `xatanh`.
//!
//! SLEEF returns infinity for `sinh` and `cosh` above 709 and for `asinh` and `acosh` above the
//! square root of the largest f64, though the results are finite up to about 710.48 and to the
//! largest f64. Here those ranges take `e^x / 2` and `ln(2x)`, which are exact to below an ulp
//! there.

use cubecl_core::{self as cubecl, prelude::*};

use super::base::{LN2_HI, LN2_LO, SQRT_DBL_MAX, mul_sign, sign_bit};
use super::double_double::{
    add_dd, add2_df, add2_ff, dd, div_dd, mul_dd, mul_df, normalize, recip_f, scale, sqrt_d, square,
};
use super::exponential::{exp, expk, expm1k, logk2};

/// `sinh` and `cosh` take `e^x / 2` from here on, where `e^-x` is below an ulp.
const HYPERBOLIC_LARGE: f64 = 709.0;
/// `e^x / 2` overflows from about 710.476 on; past this the double-double argument would too.
const HALF_EXP_MAX: f64 = 710.5;
/// `tanh` rounds to 1 from here on.
const TANH_ONE: f64 = 18.714_973_875;

#[cube]
pub(super) fn sinh(x: f64) -> f64 {
    let e = expm1k(x.abs());
    let y = (e + 2.0) / (e + 1.0) * (0.5 * e);
    let y = select(x.abs() > HYPERBOLIC_LARGE, half_exp(x.abs()), y);
    select(x.is_nan(), f64::NAN, mul_sign(y, x))
}

#[cube]
pub(super) fn cosh(x: f64) -> f64 {
    let e = exp(x.abs());
    let y = fma(0.5, e, 0.5 / e);
    let y = select(x.abs() > HYPERBOLIC_LARGE, half_exp(x.abs()), y);
    select(x.is_nan(), f64::NAN, y)
}

/// `e^x / 2` for a large `x`, as `e^(x - ln 2)` with the subtraction in double-double.
#[cube]
fn half_exp(x: f64) -> f64 {
    let shifted = expk(add2_df(dd(-LN2_HI, -LN2_LO), x));
    select(x > HALF_EXP_MAX, f64::INFINITY, shifted)
}

#[cube]
pub(super) fn tanh(x: f64) -> f64 {
    let d = expm1k(2.0 * x.abs());
    let y = d / (2.0 + d);
    let y = select(x.abs() > TANH_ONE || y.is_nan(), 1.0, y);
    select(x.is_nan(), f64::NAN, mul_sign(y, x))
}

#[cube]
pub(super) fn asinh(x: f64) -> f64 {
    let y = x.abs();
    let large = y > 1.0;

    // sqrt(x^2 + 1), from 1/x for a large x so that the square does not overflow.
    let r = recip_f(x);
    let d = dd(select(large, r.hi, y), select(large, r.lo, 0.0));
    let d = sqrt_d(add2_df(square(d), 1.0));
    let scaled = mul_df(d, y);
    let d = dd(
        select(large, scaled.hi, d.hi),
        select(large, scaled.lo, d.lo),
    );

    let d = logk2(normalize(add2_df(d, x)));
    let y = d.hi + d.lo;

    let y = select(x.abs() > SQRT_DBL_MAX, mul_sign(log_twice(x.abs()), x), y);
    let y = select(x.is_nan(), f64::NAN, y);
    select(x == 0.0 && sign_bit(x), -0.0, y)
}

#[cube]
pub(super) fn acosh(x: f64) -> f64 {
    let root = mul_dd(sqrt_d(add2_ff(x, 1.0)), sqrt_d(add2_ff(x, -1.0)));
    let d = logk2(add2_df(root, x));
    let y = d.hi + d.lo;

    let y = select(x > SQRT_DBL_MAX, log_twice(x), y);
    let y = select(x == 1.0, 0.0, y);
    select(x < 1.0 || x.is_nan(), f64::NAN, y)
}

#[cube]
pub(super) fn atanh(x: f64) -> f64 {
    let y = x.abs();
    let d = logk2(div_dd(add2_ff(1.0, y), add2_ff(1.0, -y)));
    let half = scale(d, 0.5);
    let inner = select(y == 1.0, f64::INFINITY, half.hi + half.lo);
    let y = select(y > 1.0, f64::NAN, inner);
    let y = mul_sign(y, x);
    let y = select(x.is_inf() || y.is_nan(), f64::NAN, y);
    select(x.is_nan(), f64::NAN, y)
}

/// `ln(2x)` for a large positive `x`, where it is `asinh(x)` and `acosh(x)` to below an ulp.
#[cube]
fn log_twice(x: f64) -> f64 {
    let d = add_dd(logk2(dd(x, 0.0)), dd(LN2_HI, LN2_LO));
    select(x.is_inf(), f64::INFINITY, d.hi + d.lo)
}
