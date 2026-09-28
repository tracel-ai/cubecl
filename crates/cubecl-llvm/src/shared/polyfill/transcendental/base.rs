use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Polynomial coefficients are ordered from the constant term up.
#[cube]
pub(crate) fn horner<N: Size, const D: usize>(
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

/// Mantissa bits reserved for exact integer multiplication during range reduction.
const CARRIED_BITS: u32 = 12;

pub(crate) const fn leading_part(value: f64) -> f32 {
    let head = value as f32;
    f32::from_bits(head.to_bits() & !((1u32 << CARRIED_BITS) - 1))
}

pub(crate) const fn trailing_part(value: f64) -> f32 {
    (value - leading_part(value) as f64) as f32
}

#[cfg(test)]
pub(crate) fn worst_relative_error(
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

#[cfg(test)]
pub(crate) fn evaluate(coefficients: &[f32], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |total, c| total * x + *c as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

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
