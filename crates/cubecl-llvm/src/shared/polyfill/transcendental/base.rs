use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// A polynomial in `x`, its coefficients given from the constant term up.
///
/// The loop is unrolled, so what reaches the target is the same chain of fused
/// multiply-adds writing it out by hand would give, with the nesting the wrong way round
/// for a reader.
#[cube]
pub(super) fn horner<N: Size, const D: usize>(
    x: Vector<f32, N>,
    #[comptime] coefficients: [f32; D],
) -> Vector<f32, N> {
    let mut total = Vector::new(coefficients[comptime![D - 1]]);

    #[unroll]
    for i in 1..D {
        total = fma(total, x, Vector::new(coefficients[comptime![D - 1 - i]]));
    }

    total
}

/// The number of mantissa bits a leading Cody-Waite part gives up.
///
/// What is left has to absorb the integer it gets multiplied by, and every reduction here
/// multiplies by a quadrant count or an exponent, neither of which exceeds twelve bits.
const CARRIED_BITS: u32 = 12;

/// The leading part of `value`, with the low mantissa bits cleared so that multiplying it
/// by an integer of up to [`CARRIED_BITS`] bits is exact.
///
/// A range reduction subtracts this from an argument of the same size and keeps the
/// difference. Rounding in that product would land directly on the digits the difference
/// is made of, which is the whole reason the constant is split at all.
pub(super) const fn leading_part(value: f64) -> f32 {
    let head = value as f32;
    f32::from_bits(head.to_bits() & !((1u32 << CARRIED_BITS) - 1))
}

/// What [`leading_part`] left behind, to the accuracy an `f32` can hold.
pub(super) const fn trailing_part(value: f64) -> f32 {
    (value - leading_part(value) as f64) as f32
}

/// The worst relative error of `approximation` against `exact` over `[from, to]`.
///
/// Dense enough that a mistyped digit cannot hide between two samples: the polynomials
/// here are smooth over their intervals, so an error the sweep misses is smaller than one
/// it finds.
#[cfg(test)]
pub(super) fn worst_relative_error(
    from: f64,
    to: f64,
    exact: impl Fn(f64) -> f64,
    approximation: impl Fn(f64) -> f64,
) -> f64 {
    const SAMPLES: usize = 100_000;

    (0..=SAMPLES)
        .map(|i| {
            let x = from + (to - from) * i as f64 / SAMPLES as f64;
            let truth = exact(x);
            if truth == 0.0 {
                0.0
            } else {
                ((approximation(x) - truth) / truth).abs()
            }
        })
        .fold(0.0, f64::max)
}

/// A polynomial in `x` with `coefficients` from the constant term up, in double precision.
///
/// The host mirror of [`horner`], for tests that ask what the checked-in coefficients are
/// worth without a target in the way.
#[cfg(test)]
pub(super) fn evaluate(coefficients: &[f32], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |total, c| total * x + *c as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A leading part multiplies exactly by any integer a range reduction can produce.
    ///
    /// This is the property the split exists for. Lose it and the reduction rounds away
    /// the digits of a difference it is about to keep, which shows up as an error that
    /// grows with the argument rather than as a wrong constant.
    #[test]
    fn a_leading_part_multiplies_exactly() {
        for value in [core::f64::consts::LN_2, core::f64::consts::FRAC_PI_2] {
            let head = leading_part(value);
            for multiplier in 1..=(1 << CARRIED_BITS) {
                let product = multiplier as f32 * head;
                assert_eq!(
                    product as f64,
                    multiplier as f64 * head as f64,
                    "{multiplier} * {head} rounded"
                );
            }
        }
    }

    /// A split loses no more than the rounding of its own last part.
    ///
    /// What is left over is the floor on how small a reduced argument can get and still
    /// mean anything, so it is worth knowing that it is as small as the parts allow
    /// rather than merely small.
    #[test]
    fn a_split_loses_only_its_last_rounding() {
        for (value, parts) in [
            (core::f64::consts::LN_2, 2),
            (core::f64::consts::FRAC_PI_2, 3),
        ] {
            let mut rest = value;
            let mut total = 0.0;
            for _ in 0..parts - 1 {
                let head = leading_part(rest) as f64;
                total += head;
                rest -= head;
            }
            let last = rest as f32;
            total += last as f64;

            let residual = (value - total).abs();
            let spacing = (f32::from_bits(last.abs().to_bits() + 1) - last.abs()) as f64;
            assert!(
                residual < spacing,
                "{value} left {residual} after {parts} parts, more than the last part's {spacing}"
            );
        }
    }
}
