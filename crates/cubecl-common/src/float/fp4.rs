use core::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
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
    pub const MAX: e2m1 = e2m1(0x7);
    /// Minimum representable value
    pub const MIN: e2m1 = e2m1(0xf);

    /// Constructs a [`e2m1`] value from the raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> e2m1 {
        e2m1(bits)
    }

    /// Constructs a [`e2m1`] value from a 32-bit floating point value.
    ///
    /// This operation is lossy: values past ±6 saturate to ±6, NaN becomes +6, and every other
    /// value rounds to the nearest representable value, ties to even.
    #[inline]
    #[must_use]
    pub const fn from_f32(value: f32) -> e2m1 {
        Self::from_f64(value as f64)
    }

    /// Constructs a [`e2m1`] value from a 64-bit floating point value.
    ///
    /// This operation is lossy: values past ±6 saturate to ±6, NaN becomes +6, and every other
    /// value rounds to the nearest representable value, ties to even.
    #[inline]
    #[must_use]
    pub const fn from_f64(value: f64) -> e2m1 {
        e2m1(f64_to_e2m1_bits(value))
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
        const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let magnitude = MAGNITUDES[(self.0 & 0x7) as usize];
        if self.0 & 0x8 != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

// Adapted from `f64_to_fp4` in float4 0.2 (https://github.com/EricLBuehler/float4), itself based on
// NVIDIA's `cuda_fp4.hpp`, under the MIT License:
//
// Copyright (c) 2024 Eric Buehler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
// OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
/// The E2M1 bits of `x` in the low nibble, rounded to nearest even and saturated to ±6, with NaN
/// saturating to +6.
const fn f64_to_e2m1_bits(x: f64) -> u8 {
    const EXP_BIAS: i16 = 1;
    const SIGNIFICAND_BITS: u64 = 2;
    const MANTISSA_MASK: u8 = 0x1;
    const MIN_DENORM_HALF: u64 = 0x3FD0_0000_0000_0000; // 2^-2
    const OVERFLOW_THRESHOLD: u64 = 0x4018_0000_0000_0000; // 6.0
    const MAX_NORM: u8 = 0x7;
    const MIN_NORM: u64 = 0x3FF0_0000_0000_0000; // 1.0
    const F64_INF_BITS: u64 = 0x7FF0_0000_0000_0000;

    let xbits = x.to_bits();
    let absx = xbits & 0x7FFF_FFFF_FFFF_FFFF;
    let mut sign = ((xbits >> 63) as u8) << 3;
    let exp = (((xbits >> 52) & 0x7FF) as i16 - 1023 + EXP_BIAS) as i8;
    let mantissa = ((xbits >> (53 - SIGNIFICAND_BITS)) as u8) & MANTISSA_MASK;
    // Half an E2M1 ulp, in the f64 mantissa field.
    let half_ulp: u64 = 1 << (53 - SIGNIFICAND_BITS - 1);

    let mut bits: u8;
    if absx <= MIN_DENORM_HALF {
        bits = 0;
    } else if absx > OVERFLOW_THRESHOLD {
        if absx > F64_INF_BITS {
            sign = 0;
        }
        bits = MAX_NORM;
    } else if absx >= MIN_NORM {
        bits = ((exp as u8) << (SIGNIFICAND_BITS - 1)) | mantissa;
        let round = xbits & ((half_ulp << 1) - 1);
        if round > half_ulp || (round == half_ulp && (mantissa & 1) != 0) {
            bits = bits.wrapping_add(1);
        }
    } else {
        let shift = if exp >= 1 { 0 } else { (1 - exp) as u8 };
        bits = (mantissa | (1 << (SIGNIFICAND_BITS - 1))) >> shift;
        let round_mask = (half_ulp << (shift as u64 + 1)) - 1;
        let round = (xbits | (1 << 52)) & round_mask;
        if round > (half_ulp << shift) || (round == (half_ulp << shift) && (bits & 1) != 0) {
            bits = bits.wrapping_add(1);
        }
    }
    bits | sign
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

    /// Retrieve the stored bits from an e2m1x2
    pub fn to_bits(self) -> u8 {
        self.0
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_converts_to_its_value_and_back() {
        for code in 0..16u8 {
            let value = e2m1::from_bits(code).to_f32();
            assert_eq!(
                e2m1::from_f32(value).to_bits(),
                code,
                "{code:#x} is {value}"
            );
        }
    }

    #[test]
    fn values_round_to_the_nearest_code_ties_to_even_and_saturate() {
        for (value, expected) in [
            (0.25, 0.0),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
            (0.3, 0.5),
            (2.4, 2.0),
            (100.0, 6.0),
            (-100.0, -6.0),
            (f32::NAN, 6.0),
        ] {
            assert_eq!(e2m1::from_f32(value).to_f32(), expected, "{value}");
        }
    }
}
