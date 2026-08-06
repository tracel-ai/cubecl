use core::fmt::{Display, Formatter};

/// An exact ratio of two integers, reduced to lowest terms on construction.
///
/// Unlike a float, a `Ratio` compares and hashes exactly: two ratios built
/// from different numerator/denominator pairs are equal whenever they denote
/// the same fraction (`Ratio::new(1, 2) == Ratio::new(2, 4)`), with no
/// rounding or bit-pattern comparison involved. This makes it a good fit for
/// comptime kernel parameters that are always derived from integers (tensor
/// shapes, tile sizes, and the like).
///
/// For a value that is not exactly the ratio of two integers, use
/// [`ComptimeFloat`](crate::ComptimeFloat) instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ratio {
    numerator: usize,
    denominator: usize,
}

impl Ratio {
    /// Create a new [`Ratio`], reduced to lowest terms.
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is zero.
    pub fn new(numerator: usize, denominator: usize) -> Self {
        assert!(denominator != 0, "ratio denominator must not be zero");
        let divisor = gcd(numerator, denominator);
        Self {
            numerator: numerator / divisor,
            denominator: denominator / divisor,
        }
    }

    /// The reduced numerator.
    pub fn numerator(self) -> usize {
        self.numerator
    }

    /// The reduced denominator.
    pub fn denominator(self) -> usize {
        self.denominator
    }

    /// Convert to an [`f32`].
    pub fn as_f32(self) -> f32 {
        self.numerator as f32 / self.denominator as f32
    }

    /// Convert to an [`f64`].
    pub fn as_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

impl Display for Ratio {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        // A zero numerator still reduces to 0/1; a zero denominator is
        // rejected by `Ratio::new` before `gcd` is reached.
        return a.max(b).max(1);
    }
    let (mut a, mut b) = (a, b);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn reduces_to_lowest_terms() {
        let r = Ratio::new(2, 4);
        assert_eq!(r.numerator(), 1);
        assert_eq!(r.denominator(), 2);
    }

    #[test]
    fn equal_fractions_are_equal() {
        assert_eq!(Ratio::new(1, 2), Ratio::new(2, 4));
        assert_eq!(Ratio::new(3, 9), Ratio::new(1, 3));
    }

    #[test]
    fn zero_numerator_reduces_to_zero_over_one() {
        let r = Ratio::new(0, 5);
        assert_eq!(r.numerator(), 0);
        assert_eq!(r.denominator(), 1);
    }

    #[test]
    #[should_panic(expected = "denominator must not be zero")]
    fn zero_denominator_panics() {
        Ratio::new(1, 0);
    }

    #[test]
    fn conversions_are_accurate() {
        let r = Ratio::new(1, 4);
        assert_eq!(r.as_f32(), 0.25);
        assert_eq!(r.as_f64(), 0.25);
    }

    #[test]
    fn display_shows_reduced_form() {
        let r = Ratio::new(6, 8);
        assert_eq!(format!("{}", r), "3/4");
    }
}
