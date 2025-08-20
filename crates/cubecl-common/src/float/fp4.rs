use core::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
use float4::F4E2M1;
use num_traits::{NumCast, ToPrimitive};

/// A 4-bit floating point type with 2 exponent bits and 1 mantissa bit.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, PartialEq, PartialOrd)]
pub struct e2m1(u8);

/// A 4-bit floating point type with 2 exponent bits and 1 mantissa bit. Packed with two elements
/// per value, to allow for conversion to/from bytes. Care must be taken to ensure the shape is
/// adjusted appropriately.
///
/// [`Minifloat`]: https://en.wikipedia.org/wiki/Minifloat
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Default, Zeroable, Pod, PartialEq, PartialOrd)]
pub struct e2m1x2(u8);

impl e2m1 {
    /// Maximum representable value
    pub const MAX: f64 = 6.0;
    /// Minimum representable value
    pub const MIN: f64 = -6.0;

    /// Constructs a [`e2m1`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> e2m1 {
        e2m1(bits)
    }

    /// Constructs a [`e2m1`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy. If the 32-bit value is too large to fit, ±∞ will result. NaN values
    /// are preserved. Subnormal values that are too tiny to be represented will result in ±0. All
    /// other values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> e2m1 {
        Self::from_f64(value as f64)
    }

    /// Constructs a [`e2m1`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy. If the 64-bit value is to large to fit, ±∞ will result. NaN values
    /// are preserved. 64-bit subnormal values are too tiny to be represented and result in ±0.
    /// Exponents that underflow the minimum exponent will result in subnormals or ±0. All other
    /// values are truncated and rounded to the nearest representable value.
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> e2m1 {
        e2m1(F4E2M1::from_f64(value).to_bits())
    }

    /// Converts a [`e2m1`] into the underlying bit representation.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Converts a [`e2m1`] value into an [`f32`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f32`].
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Converts a [`e2m1`] value into an [`f64`] value.
    ///
    /// This conversion is lossless as all values can be represented exactly in [`f64`].
    #[inline]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        F4E2M1::from_bits(self.0).to_f64()
    }
}

impl Neg for e2m1 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_f32(self.to_f32().neg())
    }
}

impl Mul for e2m1 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulAssign for e2m1 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for e2m1 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl DivAssign for e2m1 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Add for e2m1 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddAssign for e2m1 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for e2m1 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl SubAssign for e2m1 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ToPrimitive for e2m1 {
    fn to_i64(&self) -> Option<i64> {
        Some(e2m1::to_f32(*self) as i64)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(e2m1::to_f64(*self) as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(e2m1::to_f32(*self))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(e2m1::to_f64(*self))
    }
}

impl NumCast for e2m1 {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(Self::from_f32(n.to_f32()?))
    }
}

impl Display for e2m1 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", e2m1::to_f32(*self))
    }
}

impl Debug for e2m1 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl e2m1x2 {
    /// Create a new e2m1x2 from bits
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Create a slice of packed fp4 values from a slice of f32s
    pub fn from_f32_slice(f32s: &[f32]) -> alloc::vec::Vec<e2m1x2> {
        let mut out = alloc::vec![e2m1x2(0); f32s.len().div_ceil(2)];
        for (i, chunk) in f32s.chunks(2).enumerate() {
            let mut chunk = chunk.iter().copied();
            let a = chunk.next().unwrap_or_default();
            let b = chunk.next().unwrap_or_default();

            let a = e2m1::from_f32(a).0 & 0x0F;
            let b = (e2m1::from_f32(b).0 << 4) & 0xF0;
            out[i] = e2m1x2(a | b);
        }
        out
    }
}

impl Debug for e2m1x2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let a = e2m1::from_bits(self.0 & 0xF).to_f32();
        let b = e2m1::from_bits((self.0 >> 4) & 0xF).to_f32();
        f.debug_tuple("e2m1x2").field(&a).field(&b).finish()
    }
}
