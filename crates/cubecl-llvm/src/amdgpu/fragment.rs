//! What the AMDGPU matrix lowerings share, WMMA and MFMA alike: the element types they name,
//! the 16x16 output every form they emit writes, how a tile deeper than one instruction is cut
//! into instructions, and the slices of a fragment that feed each one.

use crate::prelude::*;
use cubecl_core::ir::{ElemType, FloatKind, IntKind, UIntKind, types::MatrixShape};

pub const F16: ElemType = ElemType::Float(FloatKind::F16);
pub const BF16: ElemType = ElemType::Float(FloatKind::BF16);
pub const F32: ElemType = ElemType::Float(FloatKind::F32);
pub const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);
pub const E5M2: ElemType = ElemType::Float(FloatKind::E5M2);
pub const I8: ElemType = ElemType::Int(IntKind::I8);
pub const U8: ElemType = ElemType::UInt(UIntKind::U8);
pub const I32: ElemType = ElemType::Int(IntKind::I32);

/// The output every form either lowering emits writes.
pub const INSTRUCTION_M: usize = 16;
pub const INSTRUCTION_N: usize = 16;

/// How many instructions deep the tiles either lowering offers are: one, or two in a row.
pub const TILE_DEPTHS: [usize; 2] = [1, 2];

/// The bits of a `bf16` lane, which the intrinsics declare as an integer of this width.
pub const HALF_BITS: u32 = 16;

/// The bits of an 8-bit code.
pub const BYTE_BITS: u32 = 8;

/// The bits of a 32-bit register.
pub const WORD_BITS: u32 = 32;

/// How many instructions `instruction_k` deep a tile takes, where its output is one
/// instruction's and its depth a whole number of them.
pub fn instruction_steps(shape: MatrixShape, instruction_k: usize) -> Option<usize> {
    let MatrixShape { m, n, k } = shape;
    let output = m == INSTRUCTION_M && n == INSTRUCTION_N;
    let depth = k > 0 && k.is_multiple_of(instruction_k);
    (output && depth).then_some(k / instruction_k)
}

/// AMD names `e4m3` `fp8` and `e5m2` `bf8`.
pub fn fp8_name(elem: ElemType) -> &'static str {
    match elem {
        E4M3 => "fp8",
        E5M2 => "bf8",
        other => unreachable!("{other:?} is no 8-bit float matrix element"),
    }
}

/// The `step`th run of `width` lanes of `fragment`: what one instruction of a deeper tile reads.
pub fn fragment_slice(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    step: usize,
    width: usize,
) -> Value {
    shuffle(
        ctx,
        rw,
        fragment,
        (0..width).map(|i| (step * width + i) as i32).collect(),
    )
}

/// `fragment`'s lanes in the order `mask` names.
pub fn shuffle(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    mask: Vec<i32>,
) -> Value {
    let op = llvm::ShuffleVectorOp::new(ctx, fragment, fragment, mask);
    insert(ctx, rw, &op)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(m: usize, n: usize, k: usize) -> MatrixShape {
        MatrixShape { m, n, k }
    }

    const WMMA_K: usize = 16;
    const MFMA_INT8_K: usize = 32;

    #[test]
    fn a_tile_deeper_than_the_instruction_is_several_of_them() {
        assert_eq!(instruction_steps(shape(16, 16, 16), WMMA_K), Some(1));
        assert_eq!(instruction_steps(shape(16, 16, 32), WMMA_K), Some(2));
    }

    #[test]
    fn a_tile_that_does_not_divide_has_no_lowering() {
        assert_eq!(instruction_steps(shape(16, 16, 24), WMMA_K), None);
        assert_eq!(instruction_steps(shape(16, 16, 8), WMMA_K), None);
        assert_eq!(instruction_steps(shape(16, 16, 0), WMMA_K), None);
        assert_eq!(instruction_steps(shape(16, 16, 16), MFMA_INT8_K), None);
    }

    /// Every instruction writes a 16x16 output, so a tile of any other has no lowering.
    #[test]
    fn a_tile_of_another_output_shape_has_no_lowering() {
        assert_eq!(instruction_steps(shape(32, 8, 16), WMMA_K), None);
        assert_eq!(instruction_steps(shape(16, 8, 32), WMMA_K), None);
        assert_eq!(instruction_steps(shape(32, 32, 16), WMMA_K), None);
    }
}
