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
        e2m1(f64_to_fp4(value))
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
        fp4_to_f64(self.0)
    }
}

// Copied from `f64_to_fp4` and `fp4_to_f64` in `src/cvt.rs` of float4 0.2.0
// (https://crates.io/crates/float4/0.2.0), published from
// https://github.com/EricLBuehler/float4/blob/bf31de14a4a79da70ec5cf668ec8091bba4685ee/src/cvt.rs
// and itself based on NVIDIA's `cuda_fp4.hpp`; one marked line differs. Under the MIT License:
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
/// Convert a host f64 to raw FP4 bits.
/// Returns the 4-bit value in the low nibble of the byte.
/// Uses round-to-nearest-even rounding mode as required by the MXFP4 specification.
pub(crate) const fn f64_to_fp4(x: f64) -> u8 {
    // ---------------------------------------------------------------------
    //  Constants for the E2M1 interpretation
    // ---------------------------------------------------------------------
    let (
        fp4_exp_bias,
        fp4_significand_bits,
        fp4_mantissa_mask,
        fp4_mindenorm_o2,
        fp4_overflow_threshold,
        fp4_maxnorm,
        fp4_minnorm,
    ) = (
        1u16,                     // bias
        2u64,                     // implicit 1 + 1 mantissa bit
        0x1u8,                    // mask for explicit mantissa bit
        0x3FD0_0000_0000_0000u64, // min denorm / 2  (2⁻²)
        0x4018_0000_0000_0000u64, // overflow thresh (6.0)
        0x7u8,                    // 0b0_111  (sign|exp|mant)
        0x3FF0_0000_0000_0000u64, // min norm (2⁰)
    );

    // --- Bit material from the source double --------------------------------
    let xbits = x.to_bits();
    let absx = xbits & 0x7FFF_FFFF_FFFF_FFFFu64;
    const DP_INF_BITS: u64 = 0x7FF0_0000_0000_0000u64;

    // Sign goes into bit 3.
    let mut sign = ((xbits >> 63) as u8) << 3;

    // Extract unbiased exponent and adapt bias for FP4.
    let exp_field = ((xbits >> 52) & 0x7FF) as u16;
    // Need to handle this as signed arithmetic to properly handle negative exponents
    let exp = (exp_field as i16 - 1023 + fp4_exp_bias as i16) as i8;

    // Mantissa shifted/truncated to target width (1 explicit bit here).
    let mantissa = ((xbits >> (53 - fp4_significand_bits)) as u8) & fp4_mantissa_mask;

    // ½-ULP of FP4 expressed in the *double* mantissa field.
    let fp4_dp_half_ulp: u64 = 1u64 << (53 - fp4_significand_bits - 1);

    // ------------------------------------------------------------------------
    //  Classify and round
    // ------------------------------------------------------------------------
    let mut res: u8;

    if absx <= fp4_mindenorm_o2 {
        // Zero or underflow → +0 (sign only retained via `sign` later).
        res = 0;
    } else if absx > fp4_overflow_threshold {
        // Overflow or NaN → saturate to FP4_MAXNORM (sign cleared for NaN).
        if absx > DP_INF_BITS {
            sign = 0; // NaN → positive.
        }
        res = fp4_maxnorm;
    } else if absx >= fp4_minnorm {
        // Normal number ------------------------------------------------------
        res = ((exp as u8) << (fp4_significand_bits - 1)) | mantissa;

        let round = xbits & ((fp4_dp_half_ulp << 1) - 1);
        // Round-to-nearest-even
        let halfway = fp4_dp_half_ulp;
        if round > halfway || (round == halfway && (mantissa & 1) != 0) {
            res = res.wrapping_add(1);
        }
    } else {
        // Denormal number ----------------------------------------------------
        let shift = if exp >= 1 { 0 } else { (1 - exp) as u8 };

        // Add implicit leading 1 before shifting.
        let denorm_mant = mantissa | (1 << (fp4_significand_bits - 1));
        res = denorm_mant >> shift;

        // Round-to-nearest-even
        let round_mask = (fp4_dp_half_ulp << (shift as u64 + 1)) - 1;
        let round = (xbits | (1u64 << 52)) & round_mask;

        if round > (fp4_dp_half_ulp << shift)
            || (round == (fp4_dp_half_ulp << shift) && (res & 1) != 0)
        {
            res = res.wrapping_add(1);
        }
    }

    // Attach the sign bit and done.
    res | sign
}

/// Convert raw FP4 bits to f64.
/// Input is expected to be a 4-bit value in the low nibble of the byte.
pub(crate) fn fp4_to_f64(fp4_bits: u8) -> f64 {
    // Extract sign bit (bit 3)
    let sign = (fp4_bits >> 3) & 1;

    // Extract exponent (bits 1-2)
    let exp_bits = (fp4_bits >> 1) & 0x3;

    // Extract mantissa (bit 0)
    let mant_bit = fp4_bits & 1;

    // E2M1 format with bias 1
    let fp4_exp_bias = 1;

    if exp_bits == 0 {
        // Denormal: exponent is 0, so actual exponent is 1 - bias = 0
        // Value = (-1)^sign * 2^0 * (0.mantissa) = (-1)^sign * mantissa/2
        let value = mant_bit as f64 * 0.5;
        if sign != 0 { -value } else { value }
    } else {
        // Normal: exponent is exp_bits - bias
        let actual_exp = exp_bits as i32 - fp4_exp_bias;
        // Value = (-1)^sign * 2^actual_exp * (1.mantissa)
        let significand = 1.0 + (mant_bit as f64 * 0.5);
        // Changed from `2.0_f64.powi(actual_exp)`, which needs std; `actual_exp` is 0, 1 or 2.
        let value = significand * (1 << actual_exp) as f64;
        if sign != 0 { -value } else { value }
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
