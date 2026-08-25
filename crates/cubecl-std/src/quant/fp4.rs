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
/// The two exponent bits, once shifted down.
const EXPONENT: u32 = 0x3;
/// The single mantissa bit.
const MANTISSA: u32 = 0x1;
/// The low nibble of a byte, one `e2m1` code.
const NIBBLE: u32 = 0xF;

/// Decode one `e2m1` code per lane, held in the low nibble of each lane of `code`.
///
/// The upper bits of a lane are ignored, so a caller may hand over an unmasked field.
#[cube]
pub fn e2m1_bits_to_float<F: Numeric, N: Size>(code: Vector<u32, N>) -> Vector<F, N> {
    let exponent = (code >> Vector::new(1u32)) & Vector::new(EXPONENT);
    let mantissa = code & Vector::new(MANTISSA);

    // The mantissa bit is worth half a step in both arms, which is what lets one term serve both.
    let half = Vector::<f32, N>::cast_from(mantissa) * Vector::new(0.5f32);

    // `exp == 0` is the subnormal arm: the mantissa bit alone, stepping by 0.5.
    let subnormal = half;

    // `exp >= 1` is `(1 + m/2) * 2^(exp-1)`. The power is taken as `(1 << exp) / 2` rather than
    // `1 << (exp - 1)`: every lane evaluates both arms, and a subnormal lane shifting by `-1`
    // would be a wrapped shift on most ISAs rather than an unused value.
    let power = Vector::<f32, N>::cast_from(Vector::new(1u32) << exponent) * Vector::new(0.5f32);
    let normal = (Vector::new(1.0f32) + half) * power;

    let magnitude = select_many(exponent.equal(&Vector::new(0u32)), subnormal, normal);

    // Negating the magnitude rather than stamping the sign onto a bit pattern keeps `-0.0`
    // decoding as `-0.0`, which is what the host codec produces for code `0x8`.
    let negative = (code & Vector::new(SIGN)).equal(&Vector::new(SIGN));
    let signed = select_many(negative, -magnitude, magnitude);

    Vector::<F, N>::cast_from(signed)
}

/// Decode the `N` `e2m1` codes packed into the low `4 * N` bits of `byte`, lowest nibble first.
///
/// The storage order is the host `e2m1x2`'s: element 0 in the low nibble, element 1 in the high
/// one. Written against `N` rather than fixed at two so a wider native pack decodes the same way.
#[cube]
pub fn e2m1_packed_bits_to_float<F: Numeric, N: Size>(byte: u32) -> Vector<F, N> {
    let mut codes = Vector::<u32, N>::empty();
    #[unroll]
    for lane in 0..N::value() {
        codes.insert(lane, (byte >> (4 * lane as u32)) & NIBBLE);
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

/// The `e2m1` magnitudes, in code order. The host-side mirror of [`e2m1_bits_to_float`], for
/// tests and for callers building a lookup table.
pub const E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Decode an `e2m1` code on the host, the reference [`e2m1_bits_to_float`] is checked against.
pub fn e2m1_decode_host(code: u32) -> f32 {
    let magnitude = E2M1_MAGNITUDES[(code & 0x7) as usize];
    if code & SIGN != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Encode to an `e2m1` code on the host, the reference [`float_to_e2m1_bits`] is checked against.
pub fn e2m1_encode_host(value: f32) -> u32 {
    let magnitude = value.abs();
    let midpoints = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let mut code = 0u32;
    for (index, midpoint) in midpoints.iter().enumerate() {
        // Odd steps land on an odd code, so their tie belongs to the neighbour above; even steps
        // keep theirs. Mirrors the alternation in the kernel.
        let clears = if index % 2 == 0 {
            magnitude > *midpoint
        } else {
            magnitude >= *midpoint
        };
        if clears {
            code += 1;
        }
    }
    if value.is_sign_negative() {
        code | SIGN
    } else {
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host codec is the kernel's specification, so it has to be right on its own terms: every
    /// code round-trips through its magnitude, and every magnitude encodes back to its code.
    #[test]
    fn every_code_round_trips() {
        for code in 0..8u32 {
            let value = e2m1_decode_host(code);
            assert_eq!(
                e2m1_encode_host(value),
                code,
                "code {code} decodes to {value}, which re-encodes elsewhere"
            );
        }
    }

    /// The sign is carried, including onto the negative zero that code `0x8` names.
    #[test]
    fn the_sign_bit_survives_a_round_trip() {
        for code in 8..16u32 {
            let value = e2m1_decode_host(code);
            assert!(value.is_sign_negative(), "code {code} decoded as {value}");
            assert_eq!(e2m1_encode_host(value), code);
        }
    }

    /// Ties go to the even code. Rounding them outward instead would bias every block upward,
    /// which on a format with only eight magnitudes is a visible shift rather than a rounding
    /// detail.
    #[test]
    fn midpoints_round_to_even() {
        for (midpoint, expected) in [
            (0.25f32, 0.0f32),
            (0.75, 1.0),
            (1.25, 1.0),
            (1.75, 2.0),
            (2.5, 2.0),
            (3.5, 4.0),
            (5.0, 4.0),
        ] {
            let landed = e2m1_decode_host(e2m1_encode_host(midpoint));
            assert_eq!(landed, expected, "{midpoint} rounded to {landed}");
        }
    }

    /// Everything past the top magnitude saturates rather than wrapping or reaching a NaN code —
    /// `e2m1` has neither an infinity nor a NaN to land on.
    #[test]
    fn magnitudes_past_the_maximum_saturate() {
        for value in [6.0f32, 6.1, 100.0, f32::MAX, f32::INFINITY] {
            assert_eq!(e2m1_decode_host(e2m1_encode_host(value)), 6.0);
            assert_eq!(e2m1_decode_host(e2m1_encode_host(-value)), -6.0);
        }
    }

    /// The codec agrees with the host `e2m1` type it is standing in for, over every code. This is
    /// what makes the software path a drop-in for the CUDA intrinsic rather than a second dialect.
    #[test]
    fn the_magnitudes_match_the_host_type() {
        for code in 0..16u32 {
            let ours = e2m1_decode_host(code);
            let theirs = cubecl_common::e2m1::from_bits(code as u8).to_f32();
            assert_eq!(
                ours, theirs,
                "code {code}: {ours} vs the host type's {theirs}"
            );
        }
    }

    /// The same agreement in the other direction, which is where the two could actually part.
    /// Decoding a code has one answer; *choosing* a code does not, and the rounding this module
    /// spells out as an alternation of strict and non-strict comparisons is the same rule
    /// `e2m1::from_f32` reaches by a different route. A tensor quantized through one and read
    /// through the other only agrees while that holds.
    #[test]
    fn the_rounding_matches_the_host_type() {
        for step in -1400..=1400i32 {
            let value = step as f32 / 100.0;
            assert_eq!(
                e2m1_encode_host(value),
                cubecl_common::e2m1::from_f32(value).to_bits() as u32,
                "{value}"
            );
        }
    }
}
