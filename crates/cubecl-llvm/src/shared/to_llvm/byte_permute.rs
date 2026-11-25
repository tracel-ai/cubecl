//! The byte permute, one instruction on the GPU targets, which take a word at a time.

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
use crate::shared::to_llvm::vector::{extract_lane, insert_vector_construct};
use crate::{prelude::*, shared::to_llvm::lower_by_target};
use cubecl_core::ir::dialect::bitwise::BytePermuteOp;

lower_by_target!(
    BytePermuteOp,
    [
        "amdgpu" AmdGpu => amdgpu,
        "nvptx" Nvptx => nvptx,
    ],
    |_: &BytePermuteOp, _: &mut Context, _: &mut DialectConversionRewriter, _| -> Result<()> {
        unreachable!("the polyfill lowers a byte permute on the CPU")
    }
);

/// The permute reads its first word as the high one.
#[cfg(feature = "amdgpu")]
fn amdgpu(
    op: &BytePermuteOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    per_word(op, ctx, rw, |ctx, rw, [low, high, selectors]| {
        let word = i32_ty(ctx);
        call_intrinsic(
            ctx,
            rw,
            "llvm.amdgcn.perm",
            word,
            vec![high, low, selectors],
        )
    })
}

/// `prmt` reads a selector a nibble: each one is moved beside the one below it, and the two pairs
/// side by side.
#[cfg(feature = "nvptx")]
fn nvptx(
    op: &BytePermuteOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _info: &OperandsInfo,
) -> Result<()> {
    const PRMT: &str = "llvm.nvvm.prmt";
    const BYTES_0_AND_2: i32 = 0x20;

    per_word(op, ctx, rw, |ctx, rw, [low, high, selectors]| {
        let word = i32_ty(ctx);
        let nibble = insert_i32_const(ctx, rw, 4);
        let zero = insert_i32_const(ctx, rw, 0);
        let bytes_0_and_2 = insert_i32_const(ctx, rw, BYTES_0_AND_2);

        let moved = llvm::LShrOp::new(ctx, selectors, nibble);
        let moved = insert(ctx, rw, &moved);
        let pairs = llvm::OrOp::new(ctx, selectors, moved);
        let pairs = insert(ctx, rw, &pairs);
        let nibbles = call_intrinsic(ctx, rw, PRMT, word, vec![pairs, zero, bytes_0_and_2]);
        call_intrinsic(ctx, rw, PRMT, word, vec![low, high, nibbles])
    })
}

/// `op` as `permute` of the low, high and selector words of each lane.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn per_word(
    op: &BytePermuteOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    permute: impl Fn(&mut Context, &mut DialectConversionRewriter, [Value; 3]) -> Value,
) -> Result<()> {
    let operands = [op.low(ctx), op.high(ctx), op.selectors(ctx)];
    let ty = cube_type_to_llvm(ctx, op.result_type(ctx));
    let lanes = ty
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements() as usize);

    let permuted = match lanes {
        None => permute(ctx, rw, operands),
        Some(lanes) => per_lane(ctx, rw, ty, lanes, operands, &permute),
    };
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![permuted]);
    Ok(())
}

/// The vector of type `ty` whose each of `lanes` lanes is `permute` of that lane of `operands`.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn per_lane(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    ty: TypeHandle,
    lanes: usize,
    operands: [Value; 3],
    permute: &impl Fn(&mut Context, &mut DialectConversionRewriter, [Value; 3]) -> Value,
) -> Value {
    let words = (0..lanes)
        .map(|lane| {
            let words = operands.map(|operand| extract_lane(ctx, rw, operand, lane));
            permute(ctx, rw, words)
        })
        .collect();
    insert_vector_construct(ctx, rw, ty, words)
}
