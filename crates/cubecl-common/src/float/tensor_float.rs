use bytemuck::{Pod, Zeroable};
use core::fmt::Display;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::{NumCast, ToPrimitive};

/// A 19-bit floating point type implementing the [`tfloat32`] format.
///
/// The [`tfloat32`] floating point format is a truncated 19-bit version of the IEEE 754 standard
/// `binary32`, a.k.a [`f32`]. [`bf16`] has approximately the same dynamic range as [`f32`] but a
/// a lower precision equal to [`f16`][half::f16].
///
/// [`tfloat32`]: https://en.wikipedia.org/wiki/TensorFloat-32
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd, Debug)]
pub struct tf32(f32);

impl tf32 {
    /// Constructs a [`tf32`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u32) -> tf32 {
        tf32(f32::from_bits(bits))
    }

    /// Constructs a [`tf32`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy. If the 32-bit value is too large to fit, ±∞ will result. NaN values
    /// are preserved. Subnormal values that are too tiny to be represented will result in ±0. All
    /// other values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> tf32 {
        tf32(value)
    }

    /// Constructs a [`tf32`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy. If the 64-bit value is to large to fit, ±∞ will result. NaN values
    /// are preserved. 64-bit subnormal values are too tiny to be represented and result in ±0.
    /// Exponents that underflow the minimum exponent will result in subnormals or ±0. All other
    /// values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> tf32 {
        tf32(value as f32)
    }

    /// Converts a [`tf32`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u32 {
        f32::to_bits(self.0)
    }

    /// Converts a [`tf32`] value into an [`f32`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    #[must_use]
    pub const fn to_f32(self) -> f32 {
        self.0
    }

    /// Converts a [`tf32`] value into an [`f64`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        self.0 as f64
    }
}

impl Neg for tf32 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_f32(self.to_f32().neg())
    }
}

impl Mul for tf32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulAssign for tf32 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for tf32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl DivAssign for tf32 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Add for tf32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for tf32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for tf32 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubAssign for tf32 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ToPrimitive for tf32 {
    fn to_i64(&self) -> Option<i64> {
        Some(tf32::to_f32(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(tf32::to_f64(*self) as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(tf32::to_f32(*self))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(tf32::to_f64(*self))
    }
}

impl NumCast for tf32 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(Self::from_f32(n.to_f32()?))
    }
}

impl Display for tf32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}
