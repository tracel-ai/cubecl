use core::{
    cmp::Ordering,
    fmt::{Debug, Display},
    num::ParseFloatError,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
use float8::F8E4M3;
use num_traits::{Num, NumCast, One, ToPrimitive, Zero};

/// A 8-bit floating point type with 4 exponent bits and 3 mantissa bits.
///
/// Follows table 1 of [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433). E4M3
/// departs from IEEE 754 to widen its range: it has **no infinities**, and reserves only
/// `S.1111.111` for NaN. Reclaiming the rest of the special-value patterns is what puts its
/// maximum at 448 rather than the 240 an IEEE-style reservation would give.
///
/// `MAX`, `MIN`, `MAX_EXP` and `MANTISSA_DIGITS` are spelled out here rather than delegated to
/// `float8`, each for the reason given on the constant itself. Every other constant delegates and
/// agrees. No infinity constant is exposed, because the format has none.
///
/// See also the [minifloat overview](https://en.wikipedia.org/wiki/Minifloat).
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e4m3(u8);

impl e4m3 {
    /// Maximum representable value, `S.1111.110` = 1.75 * 2^8 = 448.
    ///
    /// Not taken from `F8E4M3::MAX`, which is 416 and one representable step low. E4M3 has no
    /// infinities and reserves only `S.1111.111` for NaN, so `0x7E` is finite. See table 1 of
    /// "FP8 Formats for Deep Learning" (arXiv:2209.05433), which also matches what `from_f32`
    /// saturates to and what `QuantValue::E4M3` bounds against.
    pub const MAX: Self = Self::from_bits(0x7E);
    /// Minimum representable value, the negation of [`MAX`](Self::MAX).
    pub const MIN: Self = Self::from_bits(0xFE);
    /// the difference between 1.0 and the next largest representable number.
    pub const EPSILON: Self = Self::from_bits(F8E4M3::EPSILON.to_bits());
    /// Minimum representable value
    pub const MIN_POSITIVE: Self = Self::from_bits(F8E4M3::MIN_POSITIVE.to_bits());
    ///Approximate number of significant digits in base 10
    pub const DIGITS: u32 = F8E4M3::DIGITS;
    /// Number of mantissa digits, the 3 stored bits plus the implicit leading one.
    ///
    /// `F8E4M3::MANTISSA_DIGITS` counts only the stored bits, where [`f32`] and [`half::f16`] both
    /// count the implicit one.
    pub const MANTISSA_DIGITS: u32 = 4;
    /// Maximum possible normal power of 10 exponent
    pub const MAX_10_EXP: i32 = F8E4M3::MAX_10_EXP;
    /// One greater than the maximum possible normal power of 2 exponent, matching [`f32::MAX_EXP`].
    ///
    /// [`MAX`](Self::MAX) is 1.75 * 2^8, so this is 9. `F8E4M3::MAX_EXP` is 7: one lower because it
    /// reserves an exponent for infinities this format does not have, and another because it does
    /// not add the one that [`f32`] and [`half::f16`] do.
    pub const MAX_EXP: i32 = 9;
    /// Minimum possible normal power of 10 exponent
    pub const MIN_10_EXP: i32 = F8E4M3::MIN_10_EXP;
    /// Minimum possible normal power of 2 exponent
    pub const MIN_EXP: i32 = F8E4M3::MIN_EXP;
    /// The radix, or base, of the floating-point representation.
    pub const RADIX: u32 = 2;
    /// nan
    pub const NAN: Self = Self::from_bits(0xFFu8);
    /// Zero
    pub const ZERO: Self = Self::from_bits(F8E4M3::ZERO.to_bits());
    /// Negative Zero
    pub const NEG_ZERO: Self = Self::from_bits(F8E4M3::NEG_ZERO.to_bits());
    /// One
    pub const ONE: Self = Self::from_bits(F8E4M3::ONE.to_bits());

    /// Constructs a [`e4m3`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> e4m3 {
        e4m3(bits)
    }

    /// Constructs a [`e4m3`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy. Values too large to fit, infinities included, saturate to
    /// ±[`MAX`](Self::MAX), since the format has no infinities of its own. NaN values are
    /// preserved. Subnormal values that are too tiny to be represented will result in ±0. All
    /// other values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> e4m3 {
        Self::from_f64(value as f64)
    }

    /// Constructs a [`e4m3`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy. Values too large to fit, infinities included, saturate to
    /// ±[`MAX`](Self::MAX), since the format has no infinities of its own. NaN values are
    /// preserved. 64-bit subnormal values are too tiny to be represented and result in ±0.
    /// Exponents that underflow the minimum exponent will result in subnormals or ±0. All other
    /// values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> e4m3 {
        e4m3(F8E4M3::from_f64(value).to_bits())
    }

    /// Converts a [`e4m3`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Converts a [`e4m3`] value into an [`f32`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// check if an [`e4m3`] value is Nan
    #[inline]
    pub fn is_nan(self) -> bool {
        F8E4M3::is_nan(&self.into())
    }

    /// Converts a [`e4m3`] value into an [`f64`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        F8E4M3::from_bits(self.0).to_f64()
    }

    /// Compares [`e4m3`] values
    pub fn total_cmp(self, other: Self) -> Ordering {
        F8E4M3::total_cmp(&self.into(), &other.into())
    }
}

impl Zero for e4m3 {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }
    #[inline]
    fn is_zero(&self) -> bool {
        [Self::ZERO, Self::NEG_ZERO].contains(self)
    }
}

impl One for e4m3 {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
    #[inline]
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
}

impl Num for e4m3 {
    type FromStrRadixErr = ParseFloatError;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix != 10 {
            return "".parse::<f32>().map(|_| unreachable!());
        }
        let val_f32 = src.parse::<f32>()?;
        Ok(Self::from_f32(val_f32))
    }
}

impl From<F8E4M3> for e4m3 {
    fn from(value: F8E4M3) -> Self {
        e4m3(value.to_bits())
    }
}

impl From<e4m3> for F8E4M3 {
    fn from(value: e4m3) -> Self {
        Self::from_bits(value.to_bits())
    }
}

impl Neg for e4m3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_f32(self.to_f32().neg())
    }
}

impl Mul for e4m3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulAssign for e4m3 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for e4m3 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl DivAssign for e4m3 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for e4m3 {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl RemAssign for e4m3 {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Add for e4m3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for e4m3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for e4m3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubAssign for e4m3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ToPrimitive for e4m3 {
    fn to_i64(&self) -> Option<i64> {
        Some(e4m3::to_f32(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(e4m3::to_f64(*self) as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(e4m3::to_f32(*self))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(e4m3::to_f64(*self))
    }
}

impl NumCast for e4m3 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(Self::from_f32(n.to_f32()?))
    }
}

impl Display for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", e4m3::to_f32(*self))
    }
}

impl Debug for e4m3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_F32: f32 = 448.0;

    /// `MAX` diverges from `F8E4M3::MAX`, so it needs checking against the encoding rather than
    /// against the crate it would otherwise mirror.
    ///
    /// Round tripping alone would not pin it, since 416 round trips too. Saturating from above is
    /// what rules out every smaller candidate, and the step past `MAX` being NaN is what rules out
    /// any larger one.
    #[test]
    fn max_is_the_largest_finite_value() {
        assert_eq!(e4m3::from_f32(MAX_F32).to_f32(), MAX_F32);
        assert_eq!(e4m3::from_f32(f32::MAX).to_f32(), MAX_F32);
        assert!(e4m3::from_bits(e4m3::MAX.to_bits() + 1).to_f32().is_nan());
    }

    #[test]
    fn min_is_max_negated() {
        assert_eq!(e4m3::MIN.to_f32(), -MAX_F32);
    }

    /// This one does delegate to `float8`, which gets E4M3's NaN set right. Pinned so that stays
    /// true, given its neighbouring constants do not.
    #[test]
    fn is_nan_agrees_with_the_conversion_for_every_encoding() {
        for bits in 0..=u8::MAX {
            let v = e4m3::from_bits(bits);
            assert_eq!(v.is_nan(), v.to_f32().is_nan(), "0x{bits:02X}");
        }
    }
}
