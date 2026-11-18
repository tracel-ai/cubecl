//! NVPTX's own minifloat conversions, a pair of values an instruction where the software codec
//! spends a handful of integer operations a value: the decode of every minifloat it converts, and
//! the encode of `f32` to fp8.

use crate::{
    nvptx::registers::{extract_lane, insert_lane, poison},
    prelude::*,
    shared::plane::bitcast,
    shared::{intrinsic::call_op, native_decode::NativeDecode, to_llvm::general::convert_float},
};
use cubecl_core::ir::nvidia::SmArch;

/// The minifloats `arch` decodes itself. The PTX ISA gives `cvt.rn.f16x2.e4m3x2` and
/// `cvt.rn.f16x2.e5m2x2` to `sm_89` on, and `cvt.rn.f16x2.e2m1x2` to the architecture-specific
/// targets of `sm_100`, `sm_110` and `sm_120` and their minor versions, which is what
/// [`SmArch::target_cpu`] names on those parts.
pub fn native_decodes(arch: &SmArch) -> &'static [NativeDecode] {
    use NativeDecode::{E2M1x2, E4M3, E5M2};
    match arch.version() {
        100..130 => &[E2M1x2, E4M3, E5M2],
        89.. => &[E4M3, E5M2],
        _ => &[],
    }
}

impl NativeDecode {
    /// The intrinsic converting a pair of this format, held in an `i16`, to two `half`s, the low
    /// byte's first.
    fn pair_to_f16x2(self) -> &'static str {
        match self {
            NativeDecode::E2M1x2 => "llvm.nvvm.e2m1x2.to.f16x2.rn",
            NativeDecode::E4M3 => "llvm.nvvm.e4m3x2.to.f16x2.rn",
            NativeDecode::E5M2 => "llvm.nvvm.e5m2x2.to.f16x2.rn",
        }
    }

    /// The intrinsic converting two `f32`s to a pair of this format in an `i16`, the second
    /// argument in the low byte (PTX `cvt.rn.satfinite.{e4m3,e5m2}x2.f32`, the first operand in the
    /// upper half).
    fn f32_pair_to(self) -> &'static str {
        match self {
            NativeDecode::E4M3 => "llvm.nvvm.ff.to.e4m3x2.rn",
            NativeDecode::E5M2 => "llvm.nvvm.ff.to.e5m2x2.rn",
            NativeDecode::E2M1x2 => unreachable!("e2m1x2 is encoded by the software codec"),
        }
    }

    /// Bytes one conversion reads: an `e2m1x2` byte holds the pair, an fp8 pair is two bytes.
    fn bytes_per_pair(self) -> usize {
        match self {
            NativeDecode::E2M1x2 => 1,
            NativeDecode::E4M3 | NativeDecode::E5M2 => 2,
        }
    }
}

/// `input`, values of `format` held a byte each (two values a byte for `e2m1x2`, the low nibble
/// first), as the floats of `ty`: decoded a pair at a time to `half`, and converted from there.
pub(crate) fn decode(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    format: NativeDecode,
    input: Value,
    ty: TypeHandle,
) -> Value {
    let bytes = input
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements() as usize);
    let byte_count = bytes.unwrap_or(1);
    let values = match format {
        NativeDecode::E2M1x2 => 2 * byte_count,
        NativeDecode::E4M3 | NativeDecode::E5M2 => byte_count,
    };
    let half: TypeHandle = FP16Type::get(ctx).into();
    let pair_ty: TypeHandle = LlvmVectorType::get(ctx, half, 2, VectorTypeKind::Fixed).into();
    let halves_ty: TypeHandle =
        LlvmVectorType::get(ctx, half, values as u32, VectorTypeKind::Fixed).into();
    let word_ty: TypeHandle = IntegerType::get(ctx, 16, Signedness::Signless).into();
    let shift = insert_int_const(ctx, rw, 16, 8);

    let mut halves = poison(ctx, rw, halves_ty);
    let mut value_index = 0;
    for first in (0..byte_count).step_by(format.bytes_per_pair()) {
        // The pair's bytes in one `i16`, the first in the low byte, a missing second as zero.
        let mut word = None;
        for at in first..(first + format.bytes_per_pair()).min(byte_count) {
            let byte = match bytes {
                Some(_) => extract_lane(ctx, rw, input, at),
                None => input,
            };
            let widen = llvm::ZExtOp::new_with_nneg(ctx, byte, word_ty, false);
            let mut part = insert(ctx, rw, &widen);
            if at > first {
                let high = llvm::ShlOp::new_with_overflow_flag(
                    ctx,
                    part,
                    shift,
                    IntegerOverflowFlagsAttr::default(),
                );
                part = insert(ctx, rw, &high);
            }
            word = Some(match word {
                None => part,
                Some(low) => {
                    let joined = llvm::OrOp::new(ctx, low, part);
                    insert(ctx, rw, &joined)
                }
            });
        }
        let word = word.expect("a pair holds at least one byte");
        let convert = call_op(ctx, format.pair_to_f16x2(), pair_ty, vec![word]);
        let pair = insert(ctx, rw, &convert);
        for half_index in 0..2 {
            if value_index < values {
                let value = extract_lane(ctx, rw, pair, half_index);
                halves = insert_lane(ctx, rw, halves, value, value_index);
                value_index += 1;
            }
        }
    }
    let halves = match bytes {
        // A lone fp8 value is one lane of the pair, not a vector of one.
        None if values == 1 => extract_lane(ctx, rw, halves, 0),
        _ => halves,
    };
    convert_float(ctx, rw, halves, ty)
}

/// `input`, `f32`s, as the fp8 codes of `format` in `ty`, a byte each: encoded a pair at a time,
/// rounded to nearest and saturating at the format's largest finite value, the first value of a
/// pair in its low byte as [`decode`] reads it back. An even count of values is the pairs' words
/// side by side, bitcast to bytes, so a register of four codes is two words joined, not four bytes
/// placed one by one.
pub(crate) fn encode(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    format: NativeDecode,
    input: Value,
    ty: TypeHandle,
) -> Value {
    let lanes = input
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements() as usize);
    let count = lanes.unwrap_or(1);
    let word_ty: TypeHandle = IntegerType::get(ctx, 16, Signedness::Signless).into();
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let lane = |ctx: &mut Context, rw: &mut DialectConversionRewriter, at: usize| match lanes {
        Some(_) => extract_lane(ctx, rw, input, at),
        None => input,
    };
    // Each pair's word, the second value paired with the first where it is missing.
    let words: Vec<Value> = (0..count)
        .step_by(2)
        .map(|first| {
            let low = lane(ctx, rw, first);
            let high = match first + 1 < count {
                true => lane(ctx, rw, first + 1),
                false => low,
            };
            let convert = call_op(ctx, format.f32_pair_to(), word_ty, vec![high, low]);
            insert(ctx, rw, &convert)
        })
        .collect();
    match (lanes, count.is_multiple_of(2)) {
        (Some(_), true) => {
            let words_ty: TypeHandle =
                LlvmVectorType::get(ctx, word_ty, (count / 2) as u32, VectorTypeKind::Fixed).into();
            let mut joined = poison(ctx, rw, words_ty);
            for (at, &word) in words.iter().enumerate() {
                joined = insert_lane(ctx, rw, joined, word, at);
            }
            bitcast(ctx, rw, joined, ty)
        }
        _ => {
            let shift = insert_int_const(ctx, rw, 16, 8);
            let byte_of = |ctx: &mut Context, rw: &mut DialectConversionRewriter, at: usize| {
                let word = words[at / 2];
                let part = match at % 2 {
                    1 => {
                        let shifted = llvm::LShrOp::new(ctx, word, shift);
                        insert(ctx, rw, &shifted)
                    }
                    _ => word,
                };
                let byte = llvm::TruncOp::new(ctx, part, byte_ty);
                insert(ctx, rw, &byte)
            };
            match lanes {
                Some(_) => {
                    let mut codes = poison(ctx, rw, ty);
                    for at in 0..count {
                        let byte = byte_of(ctx, rw, at);
                        codes = insert_lane(ctx, rw, codes, byte, at);
                    }
                    codes
                }
                None => byte_of(ctx, rw, 0),
            }
        }
    }
}
