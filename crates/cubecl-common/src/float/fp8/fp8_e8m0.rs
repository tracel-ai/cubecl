use core::fmt::Display;

use bytemuck::{Pod, Zeroable};

/// An 8-bit unsigned floating point type with 8 exponent bits and no mantissa bits.
/// Used for scaling factors.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd, Debug)]
pub struct ue8m0(u8);

impl ue8m0 {
    /// Maximum representable value
    pub const MAX: ue8m0 = ue8m0(0xfe);
    /// Minimum representable value
    pub const MIN: ue8m0 = ue8m0(0);

    /// Constructs a [`ue8m0`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> ue8m0 {
        ue8m0(bits)
    }

    /// Constructs a [`ue8m0`] value from a 32-bit floating point value.
    ///
    /// Rounds up to a power of two, as a scale is stored: values past the range saturate to its
    /// ends, a value at or below zero becomes the minimum, and NaN stays NaN.
    #[inline]
    #[must_use]
    pub fn from_f32(value: f32) -> ue8m0 {
        Self::from_f64(value as f64)
    }

    /// Constructs a [`ue8m0`] value from a 64-bit floating point value.
    ///
    /// Rounds up to a power of two, as a scale is stored: values past the range saturate to its
    /// ends, a value at or below zero becomes the minimum, and NaN stays NaN.
    #[inline]
    #[must_use]
    pub fn from_f64(value: f64) -> ue8m0 {
        if value.is_nan() {
            return ue8m0(0xff);
        }
        if value <= 0.0 {
            return Self::MIN;
        }
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
        // The base-2 log rounded up: the exponent, plus one unless the value is a power of two.
        let log2_up = exponent + i32::from(bits & ((1 << 52) - 1) != 0);
        match log2_up + 127 {
            biased if biased < 0 => Self::MIN,
            biased if biased > 0xfe => Self::MAX,
            biased => ue8m0(biased as u8),
        }
    }

    /// Converts a [`ue8m0`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Converts a [`ue8m0`] value into an [`f32`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Converts a [`ue8m0`] value into an [`f64`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        match self.0 {
            0xff => f64::NAN,
            // 2^(code - 127), a normal f64 whatever the code.
            code => f64::from_bits((code as u64 + 1023 - 127) << 52),
        }
    }
}

impl Display for ue8m0 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ue8m0({:x})", self.0)
    }
}

mod numeric {
    use num_traits::{NumCast, ToPrimitive};

    use super::*;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    impl Neg for ue8m0 {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self::from_f32(self.to_f32().neg())
        }
    }

    impl Mul for ue8m0 {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self::from_f32(self.to_f32() * rhs.to_f32())
        }
    }

    impl MulAssign for ue8m0 {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl Div for ue8m0 {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Self::from_f32(self.to_f32() / rhs.to_f32())
        }
    }

    impl DivAssign for ue8m0 {
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    impl Add for ue8m0 {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self::from_f32(self.to_f32() + rhs.to_f32())
        }
    }

    impl AddAssign for ue8m0 {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl Sub for ue8m0 {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self::from_f32(self.to_f32() - rhs.to_f32())
        }
    }

    impl SubAssign for ue8m0 {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl ToPrimitive for ue8m0 {
        fn to_i64(&self) -> Option<i64> {
            Some(ue8m0::to_f32(*self) as i64)
        }

        fn to_u64(&self) -> Option<u64> {
            Some(ue8m0::to_f64(*self) as u64)
        }

        fn to_f32(&self) -> Option<f32> {
            Some(ue8m0::to_f32(*self))
        }

        fn to_f64(&self) -> Option<f64> {
            Some(ue8m0::to_f64(*self))
        }
    }

    impl NumCast for ue8m0 {
        fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
            Some(Self::from_f32(n.to_f32()?))
        }
    }
}
