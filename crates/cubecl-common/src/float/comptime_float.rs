use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};

/// A float type whose bit representation can stand in for [`PartialEq`]/[`Eq`]/[`Hash`].
///
/// Implemented for the IEEE-754 types that need to be used as comptime kernel
/// parameters: two values are equal iff their bit patterns are equal, so `-0.0`
/// is normalized to `0.0` (they are numerically equal) but distinct NaN payloads
/// are not conflated with each other.
pub trait FloatBits: Copy + PartialEq + Debug + Display {
    /// The unsigned integer type with the same width as `Self`.
    type Bits: Copy + Eq + Hash;

    /// The bit pattern of this value.
    fn to_bits(self) -> Self::Bits;

    /// Whether this value is neither infinite nor NaN.
    fn is_finite(self) -> bool;

    /// Positive zero, used to normalize away the sign of zero.
    fn zero() -> Self;
}

macro_rules! impl_float_bits {
    ($ty:ty, $bits:ty) => {
        impl FloatBits for $ty {
            type Bits = $bits;

            fn to_bits(self) -> $bits {
                <$ty>::to_bits(self)
            }

            fn is_finite(self) -> bool {
                <$ty>::is_finite(self)
            }

            fn zero() -> Self {
                <$ty>::from_bits(0)
            }
        }
    };
}

impl_float_bits!(f32, u32);
impl_float_bits!(f64, u64);
impl_float_bits!(half::f16, u16);
impl_float_bits!(half::bf16, u16);

/// A finite float usable as a comptime kernel parameter.
///
/// Plain floats can't back a [`Hash`]/[`Eq`] key because NaN breaks reflexivity,
/// which is required for anything used as a kernel cache key (e.g. through
/// `KernelId::info`). `ComptimeFloat` closes that gap by comparing and hashing
/// bit patterns instead, after rejecting non-finite input and normalizing `-0.0`
/// to `0.0` so numerically equal values always compare equal.
///
/// This is only appropriate for values that are genuinely arbitrary floats. If
/// a value is always an exact ratio of two integers (e.g. derived from tensor
/// shapes), prefer [`Ratio`](crate::Ratio): it is exact and avoids bit-pattern
/// comparisons entirely.
#[derive(Clone, Copy, Debug)]
pub struct ComptimeFloat<F: FloatBits>(F);

/// A float value that cannot be used as a [`ComptimeFloat`] because it is
/// infinite or NaN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidComptimeFloat<F: Debug>(pub F);

impl<F: FloatBits> Display for InvalidComptimeFloat<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "invalid comptime float: {} is not finite",
            self.0
        )
    }
}

impl<F: FloatBits> core::error::Error for InvalidComptimeFloat<F> {}

impl<F: FloatBits> ComptimeFloat<F> {
    /// Wrap `val`, rejecting non-finite input.
    pub fn new(val: F) -> Result<Self, InvalidComptimeFloat<F>> {
        if !val.is_finite() {
            return Err(InvalidComptimeFloat(val));
        }
        // IEEE-754 equality treats -0.0 == 0.0, so this also normalizes -0.0.
        let val = if val == F::zero() { F::zero() } else { val };
        Ok(Self(val))
    }

    /// The wrapped value.
    pub fn get(self) -> F {
        self.0
    }
}

impl<F: FloatBits> PartialEq for ComptimeFloat<F> {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl<F: FloatBits> Eq for ComptimeFloat<F> {}

impl<F: FloatBits> Hash for ComptimeFloat<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl<F: FloatBits> Display for ComptimeFloat<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn rejects_non_finite() {
        assert!(ComptimeFloat::new(f32::NAN).is_err());
        assert!(ComptimeFloat::new(f32::INFINITY).is_err());
        assert!(ComptimeFloat::new(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn accepts_finite() {
        assert!(ComptimeFloat::new(1.5f32).is_ok());
    }

    /// A trivial FNV-1a hasher so hash-equality can be tested without `std`.
    #[derive(Default)]
    struct TestHasher(u64);

    impl Hasher for TestHasher {
        fn finish(&self) -> u64 {
            self.0
        }

        fn write(&mut self, bytes: &[u8]) {
            self.0 = bytes.iter().fold(self.0, |hash, byte| {
                (hash ^ *byte as u64).wrapping_mul(0x100000001b3)
            });
        }
    }

    fn hash_of<T: Hash>(val: &T) -> u64 {
        let mut hasher = TestHasher::default();
        val.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn negative_zero_equals_positive_zero() {
        let neg = ComptimeFloat::new(-0.0f32).unwrap();
        let pos = ComptimeFloat::new(0.0f32).unwrap();
        assert_eq!(neg, pos);
        assert_eq!(hash_of(&neg), hash_of(&pos));
    }

    #[test]
    fn distinct_values_are_not_equal() {
        let a = ComptimeFloat::new(1.0f32).unwrap();
        let b = ComptimeFloat::new(2.0f32).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn display_matches_inner_value() {
        let val = ComptimeFloat::new(3.25f32).unwrap();
        assert_eq!(format!("{}", val), "3.25");
    }
}
