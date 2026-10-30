//! NVPTX's own minifloat decode: a pair of values an instruction, where the software codec spends a
//! handful of integer operations a value.

use crate::{
    nvptx::registers::{extract_lane, insert_lane, poison},
    prelude::*,
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
