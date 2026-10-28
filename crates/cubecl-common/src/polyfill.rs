use num_traits::Float;

/// Rust's `powi` returns mathematically incorrect results on Windows for large negative exponents.
/// This is an implementation of the standard exponentiation by squaring algorithm used by compilers
/// on most platforms to implement accurate powi. Should be bit-identical to the one in LLVM's `compiler-rt`.
pub fn powi_portable<F: Float>(x: F, n: i32) -> F {
    // n == 0 → 1.0 for *any* x, including NaN, ±0, ±inf (matches IEEE-754 / Rust)
    if n == 0 {
        return F::one();
    }
    // NaN base with non-zero exponent → NaN
    if x.is_nan() {
        return F::nan();
    }

    // ±0.0 (both compare == 0.0; distinguish sign separately)
    if x.is_zero() {
        let n_odd = n % 2 != 0;
        let neg = x.is_sign_negative() && n_odd;
        return match (n > 0, neg) {
            (true, false) => F::zero(),
            (true, true) => F::neg_zero(),
            (false, false) => F::infinity(),
            (false, true) => F::neg_infinity(),
        };
    }

    // ±infinity
    if x.is_infinite() {
        let n_odd = n % 2 != 0;
        let neg = x.is_sign_negative() && n_odd;
        return match (n > 0, neg) {
            (true, false) => F::infinity(),
            (true, true) => F::neg_infinity(),
            (false, false) => F::zero(),
            (false, true) => F::neg_zero(),
        };
    }

    // General case: exponentiation by squaring.
    let mut base = x;
    let mut exp = n.unsigned_abs();
    let mut acc = F::one();

    loop {
        if !exp.is_multiple_of(2) {
            acc = base * acc;
        }
        exp /= 2;
        if exp == 0 {
            break;
        }
        base = base * base;
    }

    if n < 0 { acc.recip() } else { acc }
}
