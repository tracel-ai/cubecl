//! Integer division facts: floors toward minus infinity and the greatest common divisor.

use cubecl::prelude::*;

use super::constant::{Integer, IntegerExpand};

/// Greatest common divisor.
pub(crate) fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `n / d` rounded toward minus infinity, for numerators that may be negative.
#[cube]
pub(crate) fn floor_div(n: i32, d: i32) -> i32 {
    let q = n / d;
    select(n % d < 0, q - 1, q)
}

/// [`floor_div`] with its non-negative remainder `n - d * floor(n/d)`.
#[cube]
pub(crate) fn floor_div_rem(n: i32, d: i32) -> (i32, i32) {
    let q = floor_div(n, d);
    (q, n.minus(q.times(d)))
}
