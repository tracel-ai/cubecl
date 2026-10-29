//! NVPTX's own `e2m1x2` decode: one instruction a pair, where the software codec spends a
//! handful of integer operations and a multiply.

use crate::{
    nvptx::registers::{extract_lane, insert_lane, poison},
    prelude::*,
    shared::{intrinsic::call_op, to_llvm::general::convert_float},
};
use cubecl_core::ir::nvidia::SmArch;

/// The intrinsic for `cvt.rn.f16x2.e2m1x2`: a pair's byte, widened to `i16`, to two `half`s, the
/// low nibble first.
const E2M1X2_TO_F16X2: &str = "llvm.nvvm.e2m1x2.to.f16x2.rn";

/// Whether `arch` converts `e2m1x2` itself. The PTX ISA gives `cvt.rn.f16x2.e2m1x2` to the
/// architecture-specific targets of `sm_100`, `sm_110` and `sm_120` and their minor versions, which is
/// what [`SmArch::target_cpu`] names on those parts.
pub fn decodes_e2m1x2(arch: &SmArch) -> bool {
    (100..130).contains(&arch.version())
}

/// `input`, `e2m1x2` pairs held a byte each, as the floats of `ty`: two a byte, the low nibble
/// first, decoded to `half` and converted from there.
pub(crate) fn decode_e2m1x2(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    input: Value,
    ty: TypeHandle,
) -> Value {
    let bytes = input
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements() as usize);
    let half: TypeHandle = FP16Type::get(ctx).into();
    let pair_ty: TypeHandle = LlvmVectorType::get(ctx, half, 2, VectorTypeKind::Fixed).into();
    let halves_ty: TypeHandle = LlvmVectorType::get(
        ctx,
        half,
        2 * bytes.unwrap_or(1) as u32,
        VectorTypeKind::Fixed,
    )
    .into();
    let word_ty: TypeHandle = IntegerType::get(ctx, 16, Signedness::Signless).into();

    let mut halves = poison(ctx, rw, halves_ty);
    for byte_index in 0..bytes.unwrap_or(1) {
        let byte = match bytes {
            Some(_) => extract_lane(ctx, rw, input, byte_index),
            None => input,
        };
        let widen = llvm::ZExtOp::new_with_nneg(ctx, byte, word_ty, false);
        let word = insert(ctx, rw, &widen);
        let convert = call_op(ctx, E2M1X2_TO_F16X2, pair_ty, vec![word]);
        let pair = insert(ctx, rw, &convert);
        for half_index in 0..2 {
            let value = extract_lane(ctx, rw, pair, half_index);
            halves = insert_lane(ctx, rw, halves, value, 2 * byte_index + half_index);
        }
    }
    convert_float(ctx, rw, halves, ty)
}
