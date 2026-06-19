use core::{
    cmp::Ordering,
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
use float8::F8E4M3;
use num_traits::{NumCast, ToPrimitive};
use rand::distr::uniform::{UniformFloat, UniformSampler};

/// A 8-bit floating point type with 4 exponent bits and 3 mantissa bits.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e4m3(u8);

impl e4m3 {
    /// Maximum representable value
    pub const MAX: Self = Self::from_bits(F8E4M3::MAX.to_bits());
    /// Minimum representable value
    pub const MIN: Self = Self::from_bits(F8E4M3::MIN.to_bits());

    /// Constructs a [`e4m3`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> e4m3 {
        e4m3(bits)
    }

    /// Constructs a [`e4m3`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy. If the 32-bit value is too large to fit, ±∞ will result. NaN values
    /// are preserved. Subnormal values that are too tiny to be represented will result in ±0. All
    /// other values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> e4m3 {
        Self::from_f64(value as f64)
    }

    /// Constructs a [`e4m3`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy. If the 64-bit value is to large to fit, ±∞ will result. NaN values
    /// are preserved. 64-bit subnormal values are too tiny to be represented and result in ±0.
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
        self.0 == 0x7Fu8 || self.0 == 0xFFu8
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

/// Sampler for [`e4m3`]
#[derive(Clone, Copy, Debug)]
pub struct E4M3Sampler(UniformFloat<f32>);

impl UniformSampler for E4M3Sampler {
    type X = e4m3;

    fn new<B1, B2>(low: B1, high: B2) -> Result<Self, rand::distr::uniform::Error>
    where
        B1: rand::distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        let l = *low.borrow();
        let h = *high.borrow();

        if l.is_nan() || h.is_nan() {
            return Err(rand::distr::uniform::Error::EmptyRange);
        }
        Ok(Self(UniformFloat::new(l.to_f32(), h.to_f32())?))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Result<Self, rand::distr::uniform::Error>
    where
        B1: rand::distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distr::uniform::SampleBorrow<Self::X> + Sized,
    {
        let l = *low.borrow();
        let h = *high.borrow();

        if l.is_nan() || h.is_nan() {
            return Err(rand::distr::uniform::Error::EmptyRange);
        }
        Ok(Self(UniformFloat::new_inclusive(l.to_f32(), h.to_f32())?))
    }

    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        e4m3::from_f32(self.0.sample(rng))
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
