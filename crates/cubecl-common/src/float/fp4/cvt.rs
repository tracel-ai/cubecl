// Copied from `src/cvt.rs` of float4 0.2.0
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

//! Based on: <https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/blob/main/cuda_fp4.hpp>

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
