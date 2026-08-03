use core::{
    cmp::Ordering,
    fmt::{Debug, Display},
    num::ParseFloatError,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
use float8::F8E5M2;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

/// A 8-bit floating point type with 5 exponent bits and 2 mantissa bits.
///
/// Follows table 1 of [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433). Unlike
/// [`e4m3`](super::e4m3), E5M2 keeps IEEE 754 conventions: infinities at `S.11111.00` and NaN at
/// `S.11111.{01,10,11}`, so six of its 256 encodings are NaN.
///
/// `MAX`, `MIN`, `is_nan`, `MAX_EXP` and `MANTISSA_DIGITS` are spelled out here rather than
/// delegated to `float8`, each for the reason given on the item itself. Every other constant
/// delegates and agrees.
///
/// See also the [minifloat overview](https://en.wikipedia.org/wiki/Minifloat).
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e5m2(u8);

impl e5m2 {
    /// Maximum representable value, `S.11110.11` = 1.75 * 2^15 = 57344.
    ///
    /// Not taken from `F8E5M2::MAX`, which is 49152 and one representable step low. E5M2 follows
    /// IEEE conventions and puts its first infinity at `S.11111.00`, so `0x7B` is finite. See
    /// table 1 of "FP8 Formats for Deep Learning" (arXiv:2209.05433), which also matches what
    /// `from_f32` saturates to and what `QuantValue::E5M2` bounds against.
    pub const MAX: e5m2 = Self::from_bits(0x7B);
    /// Minimum representable value, the negation of [`MAX`](Self::MAX).
    pub const MIN: e5m2 = Self::from_bits(0xFB);
    /// the difference between 1.0 and the next largest representable number.
    pub const EPSILON: Self = Self::from_bits(F8E5M2::EPSILON.to_bits());
    /// Minimum representable value
    pub const MIN_POSITIVE: Self = Self::from_bits(F8E5M2::MIN_POSITIVE.to_bits());
    ///Approximate number of significant digits in base 10
    pub const DIGITS: u32 = F8E5M2::DIGITS;
    /// Number of mantissa digits, the 2 stored bits plus the implicit leading one.
    ///
    /// `F8E5M2::MANTISSA_DIGITS` counts only the stored bits, where [`f32`] and [`half::f16`] both
    /// count the implicit one.
    pub const MANTISSA_DIGITS: u32 = 3;
    /// Maximum possible normal power of 10 exponent
    pub const MAX_10_EXP: i32 = F8E5M2::MAX_10_EXP;
    /// One greater than the maximum possible normal power of 2 exponent, matching [`f32::MAX_EXP`].
    ///
    /// [`MAX`](Self::MAX) is 1.75 * 2^15, so this is 16. `F8E5M2::MAX_EXP` is 15, not adding the one
    /// that [`f32`] and [`half::f16`] do.
    pub const MAX_EXP: i32 = 16;
    /// Minimum possible normal power of 10 exponent
    pub const MIN_10_EXP: i32 = F8E5M2::MIN_10_EXP;
    /// Minimum possible normal power of 2 exponent
    pub const MIN_EXP: i32 = F8E5M2::MIN_EXP;
    /// The radix, or base, of the floating-point representation.
    pub const RADIX: u32 = 2;
    /// nan
    pub const NAN: Self = Self::from_bits(0xFFu8);
    /// Zero
    pub const ZERO: e5m2 = Self::from_bits(F8E5M2::ZERO.to_bits());
    /// Negative Zero
    pub const NEG_ZERO: e5m2 = Self::from_bits(F8E5M2::NEG_ZERO.to_bits());
    /// One
    pub const ONE: e5m2 = Self::from_bits(F8E5M2::ONE.to_bits());
    /// Constructs a [`e5m2`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> e5m2 {
        e5m2(bits)
    }

    /// Constructs a [`e5m2`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy. Values too large to fit, infinities included, saturate to
    /// ±[`MAX`](Self::MAX) rather than reaching this format's own infinity. NaN values are
    /// preserved. Subnormal values that are too tiny to be represented will result in ±0. All
    /// other values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> e5m2 {
        Self::from_f64(value as f64)
    }

    /// Constructs a [`e5m2`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy. Values too large to fit, infinities included, saturate to
    /// ±[`MAX`](Self::MAX) rather than reaching this format's own infinity. NaN values are
    /// preserved. 64-bit subnormal values are too tiny to be represented and result in ±0.
    /// Exponents that underflow the minimum exponent will result in subnormals or ±0. All other
    /// values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> e5m2 {
        e5m2(F8E5M2::from_f64(value).to_bits())
    }

    /// Converts a [`e5m2`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// check if an [`e5m2`] value is Nan
    ///
    /// All six `S.11111.{01,10,11}` encodings, rather than `float8`'s two.
    #[inline]
    pub fn is_nan(self) -> bool {
        [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF].contains(&self.0)
    }

    /// Converts a [`e5m2`] value into an [`f32`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Converts a [`e5m2`] value into an [`f64`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        F8E5M2::from_bits(self.0).to_f64()
    }

    /// Compares [`e5m2`] values
    #[inline]
    pub fn total_cmp(self, other: Self) -> Ordering {
        F8E5M2::total_cmp(&self.into(), &other.into())
    }
}

impl From<F8E5M2> for e5m2 {
    fn from(value: F8E5M2) -> Self {
        e5m2(value.to_bits())
    }
}

impl From<e5m2> for F8E5M2 {
    fn from(value: e5m2) -> Self {
        Self::from_bits(value.to_bits())
    }
}

impl Zero for e5m2 {
    fn zero() -> Self {
        Self::ZERO
    }

    fn is_zero(&self) -> bool {
        [Self::ZERO, Self::NEG_ZERO].contains(self)
    }
}

impl One for e5m2 {
    fn one() -> Self {
        Self::from_bits(F8E5M2::ONE.to_bits())
    }

    fn is_one(&self) -> bool {
        self == &Self::one()
    }
}

impl Num for e5m2 {
    type FromStrRadixErr = ParseFloatError;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return "".parse::<f32>().map(|_| unreachable!());
        }
        let val_f32 = src.parse::<f32>()?;
        Ok(Self::from_f32(val_f32))
    }
}

impl Neg for e5m2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_f32(self.to_f32().neg())
    }
}

impl Mul for e5m2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulAssign for e5m2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for e5m2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl DivAssign for e5m2 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for e5m2 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl RemAssign for e5m2 {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Add for e5m2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for e5m2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for e5m2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubAssign for e5m2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ToPrimitive for e5m2 {
    fn to_i64(&self) -> Option<i64> {
        Some(e5m2::to_f32(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(e5m2::to_f64(*self) as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(e5m2::to_f32(*self))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(e5m2::to_f64(*self))
    }
}

impl NumCast for e5m2 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(Self::from_f32(n.to_f32()?))
    }
}

impl Display for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", e5m2::to_f32(*self))
    }
}

impl Debug for e5m2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_F32: f32 = 57344.0;

    /// `MAX` diverges from `F8E5M2::MAX`, so it needs checking against the encoding. This format
    /// has infinities, so the step past `MAX` is one rather than NaN.
    #[test]
    fn max_is_the_largest_finite_value() {
        assert_eq!(e5m2::from_f32(MAX_F32).to_f32(), MAX_F32);
        assert_eq!(e5m2::from_f32(f32::MAX).to_f32(), MAX_F32);
        assert!(
            e5m2::from_bits(e5m2::MAX.to_bits() + 1)
                .to_f32()
                .is_infinite()
        );
    }

    #[test]
    fn min_is_max_negated() {
        assert_eq!(e5m2::MIN.to_f32(), -MAX_F32);
    }

    /// Every encoding, checked against the conversion table rather than against the rule this
    /// restates. `float8`'s own `is_nan` recognises two of the six, which is why this does not
    /// delegate.
    #[test]
    fn is_nan_agrees_with_the_conversion_for_every_encoding() {
        for bits in 0..=u8::MAX {
            let v = e5m2::from_bits(bits);
            assert_eq!(v.is_nan(), v.to_f32().is_nan(), "0x{bits:02X}");
        }
    }
}
