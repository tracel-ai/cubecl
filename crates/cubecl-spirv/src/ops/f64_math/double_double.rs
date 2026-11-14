//! Double-double arithmetic, from SLEEF `dd.h` (the FMA variants).
//!
//! The error-free steps here need each operation rounded on its own, as SPIR-V does without
//! fast-math decorations.

use cubecl_core::{self as cubecl, prelude::*};

/// The unevaluated sum `hi + lo`, with `lo` below half an ulp of `hi`.
#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub(super) struct Dd {
    pub hi: f64,
    pub lo: f64,
}

#[cube]
pub(super) fn dd(hi: f64, lo: f64) -> Dd {
    Dd { hi, lo }
}

#[cube]
pub(super) fn normalize(t: Dd) -> Dd {
    let s = t.hi + t.lo;
    dd(s, (t.hi - s) + t.lo)
}

#[cube]
pub(super) fn scale(d: Dd, s: f64) -> Dd {
    dd(d.hi * s, d.lo * s)
}

/// `x + y` for any magnitudes.
#[cube]
pub(super) fn add2_ff(x: f64, y: f64) -> Dd {
    let sum = x + y;
    let y_part = sum - x;
    dd(sum, (x - (sum - y_part)) + (y - y_part))
}

/// `x + y` for `|x.hi| >= |y|`.
#[cube]
pub(super) fn add_df(x: Dd, y: f64) -> Dd {
    let s = x.hi + y;
    dd(s, ((x.hi - s) + y) + x.lo)
}

/// `x + y` for any magnitudes.
#[cube]
pub(super) fn add2_df(x: Dd, y: f64) -> Dd {
    let sum = x.hi + y;
    let y_part = sum - x.hi;
    let error = (x.hi - (sum - y_part)) + (y - y_part);
    dd(sum, error + x.lo)
}

/// `x + y` for `|x| >= |y.hi|`.
#[cube]
pub(super) fn add_fd(x: f64, y: Dd) -> Dd {
    let s = x + y.hi;
    dd(s, ((x - s) + y.hi) + y.lo)
}

/// `x + y` for `|x.hi| >= |y.hi|`.
#[cube]
pub(super) fn add_dd(x: Dd, y: Dd) -> Dd {
    let s = x.hi + y.hi;
    dd(s, (((x.hi - s) + y.hi) + x.lo) + y.lo)
}

/// `x + y` for any magnitudes.
#[cube]
pub(super) fn add2_dd(x: Dd, y: Dd) -> Dd {
    let sum = x.hi + y.hi;
    let y_part = sum - x.hi;
    let error = (x.hi - (sum - y_part)) + (y.hi - y_part);
    dd(sum, error + (x.lo + y.lo))
}

#[cube]
pub(super) fn mul_ff(x: f64, y: f64) -> Dd {
    let s = x * y;
    dd(s, fma(x, y, -s))
}

#[cube]
pub(super) fn mul_df(x: Dd, y: f64) -> Dd {
    let s = x.hi * y;
    dd(s, fma(x.lo, y, fma(x.hi, y, -s)))
}

#[cube]
pub(super) fn mul_dd(x: Dd, y: Dd) -> Dd {
    let s = x.hi * y.hi;
    dd(s, fma(x.hi, y.lo, fma(x.lo, y.hi, fma(x.hi, y.hi, -s))))
}

#[cube]
pub(super) fn square(x: Dd) -> Dd {
    let s = x.hi * x.hi;
    dd(s, fma(x.hi + x.hi, x.lo, fma(x.hi, x.hi, -s)))
}

#[cube]
pub(super) fn div_dd(n: Dd, d: Dd) -> Dd {
    let recip = 1.0 / d.hi;
    let quotient = n.hi * recip;
    let quotient_error = fma(recip, n.hi, -quotient);
    let recip_error = fma(-d.lo, recip, fma(-d.hi, recip, 1.0));
    let lo = fma(quotient, recip_error, fma(n.lo, recip, quotient_error));
    dd(quotient, lo)
}

#[cube]
pub(super) fn recip_f(d: f64) -> Dd {
    let s = 1.0 / d;
    dd(s, s * fma(-d, s, 1.0))
}

#[cube]
pub(super) fn sqrt_d(d: Dd) -> Dd {
    let t = (d.hi + d.lo).sqrt();
    scale(mul_dd(add2_dd(d, mul_ff(t, t)), recip_f(t)), 0.5)
}
