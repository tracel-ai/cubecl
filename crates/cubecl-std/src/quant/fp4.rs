//! Software `e2m1` conversion, on `u32` bit patterns only, so a backend with no 4-bit float type
//! can still decode and encode fp4 from the bits in a word.
//!
//! This is the fp4 counterpart of `cubecl_core::post_processing::minifloat`, and it exists for the
//! same reason: `e2m1` conversion is a CUDA intrinsic and nothing else. Every other backend either
//! has no 4-bit float type at all or can only move one around, so a quantized kernel that reaches
//! for `e2m1x2::from_bits` runs on one vendor. The arithmetic below runs everywhere.
//!
//! It does not go through the general minifloat path. That one reconstructs an `f32` bit pattern
//! field by field, which is the only tractable way to cover eight exponent bits; `e2m1` has four
//! codes per sign and its eight magnitudes are `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, small enough that
//! decoding is one select over the subnormal arm and encoding is a count of the midpoints a
//! magnitude clears.

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// The sign bit of an `e2m1` code.
const SIGN: u32 = 0x8;
/// The single mantissa bit.
const MANTISSA: u32 = 0x1;
/// The low nibble of a byte, one `e2m1` code.
const NIBBLE: u32 = 0xF;
/// A code's magnitude: its exponent and its mantissa, the sign left behind.
const MAGNITUDE: u32 = 0x7;

/// Where a code's magnitude bits land in an `f32`.
///
/// Both formats lay a number out the same way — sign, exponent, mantissa, most significant
/// first — so a code's three magnitude bits are already an `f32`'s three most significant value
/// bits, in order. An `f32`'s exponent field starts at bit 23 and a code's exponent is two bits
/// wide, so the whole magnitude moves up by this one shift and the mantissa bit lands at 22.
const MAGNITUDE_SHIFT: u32 = 22;

/// The `f32` an `e2m1` exponent of zero would name, which is both the bias the normal arm adds
/// and the one non-zero subnormal magnitude.
///
/// As a bias it is `126 << 23`: a code's exponent `e` means `2^(e-1)`, and an `f32`'s field of
/// `126 + e` means the same. As a value it is `0.5` — a field of 126 with an empty mantissa —
/// which is the only magnitude the subnormal arm has besides zero. One constant serves both
/// because they are the same number for the same reason.
const EXPONENT_BIAS: u32 = 126 << 23;

/// The shift from a code's sign bit to an `f32`'s.
const SIGN_SHIFT: u32 = 28;

/// Decode one `e2m1` code per lane, held in the low nibble of each lane of `code`.
///
/// The upper bits of a lane are ignored, so a caller may hand over an unmasked field.
///
/// The decode is an assembly of the `f32`'s bits, not arithmetic over its value. `e2m1` and
/// `f32` are the same shape of number, so a code's magnitude bits are already an `f32`'s top
/// value bits and only have to be moved into place and biased — where computing `(1 + m/2) *
/// 2^(e-1)` term by term costs two integer-to-float conversions and three multiplies to reach
/// one of sixteen possible numbers. The subnormal arm is the one place the two layouts
/// genuinely disagree and the one place a select is owed.
#[cube]
pub fn e2m1_bits_to_float<F: Numeric, N: Size>(code: Vector<u32, N>) -> Vector<F, N> {
    let magnitude = code & Vector::new(MAGNITUDE);

    // `exp >= 1` is `(1 + m/2) * 2^(exp-1)`, which is what an `f32` with exponent field
    // `126 + exp` and mantissa bit `m` already means. The add cannot carry out of the exponent
    // field: `exp` is at most three, and `126 + 3` still fits it.
    let normal = (magnitude << Vector::new(MAGNITUDE_SHIFT)) + Vector::new(EXPONENT_BIAS);

    // `exp == 0` is the subnormal arm, `m * 0.5`, so its two codes are `0.0` and `0.5` where
    // the assembly above reads `0.5` and `0.75`. Both of those are the bias constant, kept or
    // cleared by the mantissa bit.
    let mantissa = code & Vector::new(MANTISSA);
    let subnormal = select_many(
        mantissa.equal(&Vector::new(MANTISSA)),
        Vector::new(EXPONENT_BIAS),
        Vector::new(0u32),
    );

    // A magnitude above one is a non-zero exponent, the mantissa bit being all that lies below.
    let bits = select_many(magnitude.greater_than(&Vector::new(MANTISSA)), normal, subnormal);

    // The sign rides as the bit it is rather than negating a magnitude. That is a shift and an
    // or against a compare, a negate and a select — and it is also the only form that reaches
    // `-0.0`, which code `0x8` names and the host codec produces.
    let sign = (code & Vector::new(SIGN)) << Vector::new(SIGN_SHIFT);

    Vector::<F, N>::cast_from(Vector::<f32, N>::reinterpret(bits | sign))
}

/// Decode the `N` `e2m1` codes packed into the low `4 * N` bits of `word`, lowest nibble first.
///
/// The storage order is the host `e2m1x2`'s: element 0 in the low nibble, element 1 in the high
/// one. Written against `N` rather than fixed at two so a wider native pack decodes the same way.
#[cube]
pub fn e2m1_packed_bits_to_float<F: Numeric, N: Size>(word: u32) -> Vector<F, N> {
    let mut codes = Vector::<u32, N>::empty();
    #[unroll]
    for lane in 0..N::value() {
        codes.insert(lane, (word >> (4 * lane as u32)) & NIBBLE);
    }
    e2m1_bits_to_float::<F, N>(codes)
}

/// Encode one `e2m1` code per lane into the low nibble of each lane, rounding to nearest with
/// ties to even and saturating at `±6`.
///
/// Ties to even is not a detail here. `e2m1`'s magnitudes are so far apart that a tie is a common
/// input rather than a rare one — `0.75` and `2.5` are both exact midpoints — and rounding them
/// all outward would bias every quantized block upward.
///
/// The rounding is expressed as a count of the midpoints the magnitude clears, which puts the
/// whole codec in comparisons and adds. The comparisons alternate strict and non-strict on
/// purpose: that is what lands each tie on the even code (`0.75 -> 1.0`, `2.5 -> 2.0`) without a
/// separate parity fixup.
#[cube]
pub fn float_to_e2m1_bits<F: Numeric, N: Size>(value: Vector<F, N>) -> Vector<u32, N> {
    let value = Vector::<f32, N>::cast_from(value);

    // The sign comes off the bit pattern rather than a comparison against zero. `-0.0` is not
    // less than zero, so a comparison calls it positive and drops it on code `0x0`, where
    // [`e2m1_bits_to_float`] and the host codec both name it `0x8`. The negative zero a decode
    // produces has to encode back to the code it came from.
    let sign_bit = Vector::new(0x8000_0000u32);
    let negative = (Vector::<u32, N>::reinterpret(value) & sign_bit).equal(&sign_bit);
    let magnitude = select_many(negative, -value, value);

    // The midpoints of {0, 0.5, 1, 1.5, 2, 3, 4, 6}, in order.
    let mut code = cleared::<N>(magnitude.greater_than(&Vector::new(0.25f32)));
    code += cleared::<N>(magnitude.greater_equal(&Vector::new(0.75f32)));
    code += cleared::<N>(magnitude.greater_than(&Vector::new(1.25f32)));
    code += cleared::<N>(magnitude.greater_equal(&Vector::new(1.75f32)));
    code += cleared::<N>(magnitude.greater_than(&Vector::new(2.5f32)));
    code += cleared::<N>(magnitude.greater_equal(&Vector::new(3.5f32)));
    code += cleared::<N>(magnitude.greater_than(&Vector::new(5.0f32)));

    // A NaN clears no threshold and encodes as zero. `e2m1` has no NaN code to carry it to, so
    // every codec has to pick something; zero is what the saturating comparisons already give.
    code | (cleared::<N>(negative) * Vector::new(SIGN))
}

/// One per lane where the lane cleared its threshold, zero elsewhere — the term
/// [`float_to_e2m1_bits`] sums to reach a code.
#[cube]
fn cleared<N: Size>(above: Vector<bool, N>) -> Vector<u32, N> {
    select_many(above, Vector::new(1u32), Vector::new(0u32))
}

#[cfg(test)]
mod tests {
    use cubecl_common::e2m1;

    /// The rounding this module implements is round-to-nearest with ties to even, which is what
    /// `e2m1` itself does. The kernel reaches it by counting cleared midpoints and `e2m1` by a
    /// different route, so the two agreeing is the specification holding rather than a tautology.
    ///
    /// Ties are not an edge case on a grid this coarse: `0.75` and `2.5` are both exact midpoints
    /// of neighbouring code points, and rounding them all outward would bias every quantized block
    /// upward by a visible amount rather than by a rounding error.
    #[test]
    fn the_midpoints_round_to_even() {
        for (midpoint, expected) in [
            (0.25f32, 0.0f32),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
        ] {
            let landed = e2m1::from_f32(midpoint).to_f32();
            assert_eq!(landed, expected, "{midpoint} rounded to {landed}");
        }
    }

    /// Everything past the top magnitude saturates rather than wrapping or reaching a NaN code —
    /// `e2m1` has neither an infinity nor a NaN to land on. The kernel's count of cleared
    /// midpoints saturates by construction, so this pins the reference it is checked against.
    #[test]
    fn magnitudes_past_the_maximum_saturate() {
        for value in [6.0f32, 6.1, 100.0, f32::MAX, f32::INFINITY] {
            assert_eq!(e2m1::from_f32(value).to_f32(), 6.0);
            assert_eq!(e2m1::from_f32(-value).to_f32(), -6.0);
        }
    }

    /// The sign is carried onto the negative zero that code `0x8` names, which is the one code
    /// telling `-0.0` from `0.0` depends on.
    #[test]
    fn the_sign_bit_survives_a_round_trip() {
        for code in 8..16u8 {
            let value = e2m1::from_bits(code).to_f32();
            assert!(value.is_sign_negative(), "code {code} decoded as {value}");
            assert_eq!(e2m1::from_f32(value).to_bits(), code);
        }
    }
}
