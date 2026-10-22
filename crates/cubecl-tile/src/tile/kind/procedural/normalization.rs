//! The divisor of a normalized separable factor run: which taps it sums, and its zero guard.

use cubecl::prelude::*;

use crate::Space;

/// What a normalized factor run sums and divides by, over the complete runs of `over`.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Normalization {
    pub taps: TapSupport,
    pub guard: DivGuard,
    pub over: Space,
}

impl Normalization {
    pub(crate) fn new(taps: TapSupport, guard: DivGuard, over: Space) -> Self {
        Normalization { taps, guard, over }
    }
}

/// How division handles a denominator too small to divide by. The default maps it to zero.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DivGuard {
    pub epsilon: f32,
    pub fallback: f32,
}

impl DivGuard {
    /// Falls back to `fallback` where the denominator's magnitude is at most `epsilon`.
    pub fn new(epsilon: f32, fallback: f32) -> Self {
        assert!(
            epsilon.is_finite() && epsilon >= 0.0,
            "DivGuard: epsilon must be finite and non-negative"
        );
        assert!(fallback.is_finite(), "DivGuard: fallback must be finite");
        DivGuard { epsilon, fallback }
    }
}

/// Which taps contribute to the sum of a normalized separable filter factor.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TapSupport {
    /// Sum only taps whose projected input sample is in bounds.
    /// The contraction must read the source window in place; a shared-memory stage is rejected.
    #[default]
    InBounds,
    /// Sum the filter's whole support, preserving the fade of zero padding at an edge.
    Whole,
}

/// A guarded reciprocal that preserves the sign of a valid denominator; NaN takes the fallback.
#[cube]
pub(crate) fn guarded_recip<E: Numeric>(d: E, #[comptime] guard: DivGuard) -> E {
    let epsilon = E::cast_from(comptime!(guard.epsilon));
    let fallback = E::cast_from(comptime!(guard.fallback));
    let valid = d.abs() > epsilon;
    let safe = select(valid, d, E::from_int(1));
    select(valid, E::from_int(1) / safe, fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_guard_accepts_finite_non_negative_thresholds() {
        DivGuard::new(0.0, 0.0);
        DivGuard::new(1.0e-7, -1.0);
    }

    #[test]
    fn division_guard_rejects_invalid_thresholds() {
        for epsilon in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| DivGuard::new(epsilon, 0.0)).is_err(),
                "epsilon {epsilon:?} should be rejected"
            );
        }
        for fallback in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                std::panic::catch_unwind(|| DivGuard::new(1.0e-7, fallback)).is_err(),
                "fallback {fallback:?} should be rejected"
            );
        }
    }
}
