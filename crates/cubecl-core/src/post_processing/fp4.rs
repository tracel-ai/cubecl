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

use cubecl_ir::{
    NamedRewrite, Scope,
    dialect::{base::OperationPtrExt, general::CastOp},
    interfaces::TypedExt,
    prelude::*,
    types::scalar::Float4E2M1x2Type,
};
use half::f16;
use pliron::r#type::TypeHandle;

use crate::post_processing::minifloat::Fp8Container;
use crate::{self as cubecl, prelude::*};

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
    let bits = select_many(
        magnitude.greater_than(&Vector::new(MANTISSA)),
        normal,
        subnormal,
    );

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

/// What a code placed as an `f16` near the bottom of its range ([`f16_pair_bits`]) is short of
/// its value by: `2^14`, the gap between an `e2m1` exponent of zero and an `f16` one.
pub const E2M1_F16_LIFT: f32 = 16384.0;

/// The `f16` pair two codes name, from a word holding one code in its low nibble and the other
/// sixteen bits up.
///
/// An `e2m1` code is already an `f16`'s top bits in order, sign, exponent, mantissa, so each
/// lands by one mask and one shift: the magnitude where an `f16`'s exponent ends, the sign on its
/// sign bit. What it lands on is the value `2^14` times too small, and exactly that: the exponent
/// field is the code's own, so a code of exponent zero is an `f16` subnormal, which is the
/// format's own subnormal arm with no select. Two codes sixteen bits apart move together, so one
/// pair costs what one code does.
#[cube]
fn f16_pair_bits(codes: u32) -> u32 {
    ((codes & 0x0007_0007) << 9) | ((codes & 0x0008_0008) << 12)
}

/// Decode the eight codes of each word of `words`, lowest nibble first, to `f16`.
///
/// One multiply for every lane lifts the placed values ([`e2m1_words_to_f16_placed`]) to their
/// values, exactly, the factor being a power of two.
#[cube]
pub fn e2m1_words_to_f16<W: Size, V: Size>(words: Vector<u32, W>) -> Vector<f16, V> {
    e2m1_words_to_f16_placed::<W, V>(words) * Vector::new(f16::new(E2M1_F16_LIFT))
}

/// The eight codes of each word of `words`, lowest nibble first, placed as `f16` each
/// [`E2M1_F16_LIFT`] short of its value, exactly: what a caller that multiplies the values by a
/// factor anyway, a block scale, takes, folding the lift into that factor rather than paying a
/// multiply a value.
///
/// A word's codes `j` and `j + 4` sit sixteen bits apart, so a word is four pairs and four
/// [`f16_pair_bits`]; the pairs land on their lanes by compile-time inserts.
#[cube]
pub fn e2m1_words_to_f16_placed<W: Size, V: Size>(words: Vector<u32, W>) -> Vector<f16, V> {
    let mut values = Vector::<f16, V>::empty();
    #[unroll]
    for w in 0..W::value() {
        let word = words.extract(w);
        #[unroll]
        for j in 0..4usize {
            let shift = comptime![4 * j as u32];
            let pair = Vector::<f16, Const<2>>::reinterpret(f16_pair_bits(word >> shift));
            values.insert(8 * w + j, pair.extract(0usize));
            values.insert(8 * w + j + 4, pair.extract(1usize));
        }
    }
    values
}

/// [`e2m1_words_to_f16`] for bytes that do not fill a word, one `e2m1x2` per lane of `bytes`:
/// the high nibble moves up to the second half of the word first, one more mask and shift a pair.
#[cube]
pub fn e2m1_bytes_to_f16<B: Size, V: Size>(bytes: Vector<u32, B>) -> Vector<f16, V> {
    let mut values = Vector::<f16, V>::empty();
    #[unroll]
    for b in 0..B::value() {
        let byte = bytes.extract(b);
        let codes = (byte & 0xF) | ((byte & 0xF0) << 12);
        let pair = Vector::<f16, Const<2>>::reinterpret(f16_pair_bits(codes));
        values.insert(2 * b, pair.extract(0usize));
        values.insert(2 * b + 1, pair.extract(1usize));
    }
    values * Vector::new(f16::new(E2M1_F16_LIFT))
}

/// The codes of the `e2m1x2` bytes in `bytes`, one a lane, low nibble first: what the `f32`
/// decode reads.
#[cube]
fn bytes_to_codes<B: Size, V: Size>(bytes: Vector<u32, B>) -> Vector<u32, V> {
    let mut codes = Vector::<u32, V>::empty();
    #[unroll]
    for b in 0..B::value() {
        let byte = bytes.extract(b);
        codes.insert(2 * b, byte & NIBBLE);
        codes.insert(2 * b + 1, (byte >> 4) & NIBBLE);
    }
    codes
}

/// The `e2m1x2` bytes two codes a lane make, low nibble first: what an encode stores.
#[cube]
fn codes_to_bytes<V: Size, B: Size>(codes: Vector<u32, V>) -> Vector<u32, B> {
    let mut bytes = Vector::<u32, B>::empty();
    #[unroll]
    for b in 0..B::value() {
        bytes.insert(
            b,
            (codes.extract(2 * b) & NIBBLE) | ((codes.extract(2 * b + 1) & NIBBLE) << 4),
        );
    }
    bytes
}

/// Bytes, one a lane, packed four to a word, lane 0 in the low byte.
#[cube]
fn bytes_to_words<B: Size, W: Size>(bytes: Vector<u32, B>) -> Vector<u32, W> {
    let mut words = Vector::<u32, W>::empty();
    #[unroll]
    for w in 0..W::value() {
        let mut word = 0u32;
        #[unroll]
        for offset in 0..4usize {
            let shift = comptime![8 * offset as u32];
            word |= (bytes.extract(4 * w + offset) & 0xFF) << shift;
        }
        words.insert(w, word);
    }
    words
}

/// Words, each four bytes, as one byte a lane, lane 0 in the low byte.
#[cube]
fn words_to_bytes<W: Size, B: Size>(words: Vector<u32, W>) -> Vector<u32, B> {
    let mut bytes = Vector::<u32, B>::empty();
    #[unroll]
    for b in 0..B::value() {
        let shift = comptime![8 * (b % 4) as u32];
        bytes.insert(b, (words.extract(b / 4) >> shift) & 0xFF);
    }
    bytes
}

define_size!(B);
define_size!(V);
define_size!(W);

pub type LowerFp4CastPass = MatchRewritePass<LowerFp4Cast>;

/// Lowers every cast from or to `e2m1x2` onto the software codec, for a backend that has no
/// native fp4 conversion.
///
/// The decode goes through `f16` pairs ([`e2m1_words_to_f16`]) where the backend has `f16`, and
/// through `f32` ([`e2m1_bits_to_float`]) where it does not: the pairs decode two codes for what
/// the `f32` path spends on one and need no select, and the backend is what knows whether `f16`
/// exists. A cast whose bytes fill whole words decodes a word at a time, codes four apart
/// sharing a pair. The encode is [`float_to_e2m1_bits`] either way.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerFp4Cast {
    /// Whether the backend computes in `f16`.
    half: bool,
    container: Fp8Container,
}

fn is_e2m1x2(ctx: &Context, ty: TypeHandle) -> bool {
    ty.scalar_ty(ctx).deref(ctx).is::<Float4E2M1x2Type>()
}

impl MatchRewrite for LowerFp4Cast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<CastOp>(ctx) {
            return false;
        }
        is_e2m1x2(ctx, op.operand(ctx, 0).get_type(ctx))
            || is_e2m1x2(ctx, op.result(ctx).get_type(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let input = op.operand(ctx, 0);
        let result_ty = op.result(ctx).get_type(ctx);
        let value = if is_e2m1x2(ctx, input.get_type(ctx)) {
            self.decode(&scope, input, result_ty)
        } else {
            self.encode(&scope, input, result_ty)
        };
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

impl LowerFp4Cast {
    /// `input`'s bytes as `u32` lanes, one byte a lane.
    fn byte_lanes(&self, scope: &Scope, input: Value, bytes: usize) -> Value {
        match self.container {
            Fp8Container::Bytes => {
                let lanes =
                    reinterpret_value(scope, input, Vector::<u8, B>::__expand_as_type(scope));
                cast_value(scope, lanes, Vector::<u32, B>::__expand_as_type(scope))
            }
            Fp8Container::Words => {
                let words = reinterpret_value(scope, input, self.words_type(scope, bytes));
                words_to_bytes::expand::<W, B>(scope, words.into()).read_value(scope)
            }
        }
    }

    /// Registers `W`, the words `bytes` bytes fill, and names their type.
    fn words_type(&self, scope: &Scope, bytes: usize) -> TypeHandle {
        assert!(
            bytes.is_multiple_of(4),
            "fp4 is packed four bytes to a u32 on this backend: vectors of {bytes} e2m1x2 lanes \
             are not supported, use a multiple of four"
        );
        scope.register_size::<W>(bytes / 4);
        Vector::<u32, W>::__expand_as_type(scope)
    }

    fn decode(&self, scope: &Scope, input: Value, result_ty: TypeHandle) -> Value {
        let ctx = scope.ctx();
        let bytes = input.vector_size(ctx);
        let values = result_ty.vector_size(ctx);
        debug_assert_eq!(values, 2 * bytes, "an e2m1x2 lane casts to two values");
        scope.register_size::<B>(bytes);
        scope.register_size::<V>(values);
        let decoded = match (self.half, bytes.is_multiple_of(4)) {
            (true, true) => {
                let words = reinterpret_value(scope, input, self.words_type(scope, bytes));
                e2m1_words_to_f16::expand::<W, V>(scope, words.into()).read_value(scope)
            }
            (true, false) => {
                let lanes = self.byte_lanes(scope, input, bytes);
                e2m1_bytes_to_f16::expand::<B, V>(scope, lanes.into()).read_value(scope)
            }
            (false, _) => {
                let lanes = self.byte_lanes(scope, input, bytes);
                let codes = bytes_to_codes::expand::<B, V>(scope, lanes.into());
                e2m1_bits_to_float::expand::<f32, V>(scope, codes).read_value(scope)
            }
        };
        cast_value(scope, decoded, result_ty)
    }

    fn encode(&self, scope: &Scope, input: Value, result_ty: TypeHandle) -> Value {
        let ctx = scope.ctx();
        let values = input.vector_size(ctx);
        let bytes = result_ty.vector_size(ctx);
        debug_assert_eq!(values, 2 * bytes, "two values cast to an e2m1x2 lane");
        scope.register_size::<B>(bytes);
        scope.register_size::<V>(values);
        let value = cast_value(scope, input, Vector::<f32, V>::__expand_as_type(scope));
        let codes = float_to_e2m1_bits::expand::<f32, V>(scope, value.into());
        let packed = codes_to_bytes::expand::<V, B>(scope, codes);
        let container = match self.container {
            Fp8Container::Bytes => cast_value(
                scope,
                packed.read_value(scope),
                Vector::<u8, B>::__expand_as_type(scope),
            ),
            Fp8Container::Words => {
                self.words_type(scope, bytes);
                bytes_to_words::expand::<B, W>(scope, packed).read_value(scope)
            }
        };
        reinterpret_value(scope, container, result_ty)
    }
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
