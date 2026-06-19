use core::{
    cmp::Ordering, fmt::{Debug, Display}, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}
};

use bytemuck::{Pod, Zeroable};
use float8::F8E4M3;
use num_traits::{NumCast, ToPrimitive};
use rand::distr::uniform::UniformSampler;

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

impl UniformSampler for e4m3 {
    type X = e4m3;

    fn new<B1, B2>(_low: B1, _high: B2) -> Result<Self, rand::distr::uniform::Error>
    where
        B1: rand::distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distr::uniform::SampleBorrow<Self::X> + Sized {
        // Ok(Self(UniformFloat::new(
        //     low.borrow().to_f32(),
        //     high.borrow().to_f32(),
        // )?))
        todo!()
    }

    fn new_inclusive<B1, B2>(_low: B1, _high: B2) -> Result<Self, rand::distr::uniform::Error>
    where
        B1: rand::distr::uniform::SampleBorrow<Self::X> + Sized,
        B2: rand::distr::uniform::SampleBorrow<Self::X> + Sized {
        todo!()
    }

    fn sample<R: rand::prelude::Rng + ?Sized>(&self, _rng: &mut R) -> Self::X {
        //Self::from_f32(self.0.sample(rng))
        todo!()
    }
}

impl From<F8E4M3> for e4m3 {
    fn from(value: F8E4M3) -> Self {
        e4m3(value.to_bits())
    }
}


impl Into<F8E4M3> for e4m3 {
    fn into(self) -> F8E4M3 {
        F8E4M3::from_bits(self.to_bits())
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
