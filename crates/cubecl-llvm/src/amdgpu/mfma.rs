//! CDNA's MFMA instructions, the matrix cores of MI200 and MI300.
//!
//! Every form this lowering emits writes a 16x16 tile over a 64-wide wavefront, laid out as
//! AMD's matrix instruction calculator gives it: lane `l` sits in row group `l / 16` at column
//! `l % 16`. Of A it holds row `l % 16` and of B column `l % 16`, each a run of
//! `instruction_k / 4` consecutive `k` starting at `(l / 16) * instruction_k / 4`. Of the
//! accumulator it holds column `l % 16`, rows `4 * (l / 16)` to `4 * (l / 16) + 3`.

use crate::{prelude::*, shared::plane::bitcast};
use cubecl_core::ir::{
    ElemType, FloatKind, IntKind,
    amd::{AmdMfma, MFMA_LANE_GROUPS},
    features::MmaConfig,
    types::MatrixShape,
};

const F16: ElemType = ElemType::Float(FloatKind::F16);
const BF16: ElemType = ElemType::Float(FloatKind::BF16);
const F32: ElemType = ElemType::Float(FloatKind::F32);
const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);
const E5M2: ElemType = ElemType::Float(FloatKind::E5M2);
const I8: ElemType = ElemType::Int(IntKind::I8);
const I32: ElemType = ElemType::Int(IntKind::I32);

/// The output every form this lowering emits writes.
pub const INSTRUCTION_M: usize = 16;
pub const INSTRUCTION_N: usize = 16;

/// Accumulator elements each lane holds: a 16x16 output over 64 lanes.
pub const ACCUMULATOR_REGISTERS: usize = 4;

/// How many instructions deep the advertised tiles are: one, or two in a row.
const TILE_DEPTHS: [usize; 2] = [1, 2];

/// The `cbsz`, `abid` and `blgp` modifiers, which broadcast and permute blocks of A and B. The
/// 16x16 single-block forms take none.
const NO_BROADCAST: i32 = 0;

/// The MFMA instruction a tile's element types select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MfmaForm {
    /// `f16` or `bf16` operands accumulating in `f32`, 16 deep.
    Half { ab: ElemType },
    /// Signed 8-bit operands accumulating in `i32`: 16 deep on CDNA2, 32 deep on CDNA3.
    Int8,
    /// 8-bit float operands, each side its own format, accumulating in `f32`, 32 deep. CDNA3
    /// only, and it reads its bytes as FNUZ ([`reads_fnuz`](Self::reads_fnuz)).
    Fp8 { a: ElemType, b: ElemType },
}

impl MfmaForm {
    /// The form multiplying `a` by `b` into `cd` on `generation`, where MFMA has one. MFMA
    /// accumulates in 32 bits only, its 8-bit integer forms take signed operands on both sides,
    /// and only CDNA3 has 8-bit float forms.
    pub fn of(generation: AmdMfma, a: ElemType, b: ElemType, cd: ElemType) -> Option<Self> {
        match (a, b, cd) {
            (F16 | BF16, _, F32) if b == a => Some(MfmaForm::Half { ab: a }),
            (I8, I8, I32) => Some(MfmaForm::Int8),
            (E4M3 | E5M2, E4M3 | E5M2, F32) if generation == AmdMfma::Cdna3 => {
                Some(MfmaForm::Fp8 { a, b })
            }
            _ => None,
        }
    }

    /// Whether the form reads its 8-bit float bytes as AMD's FNUZ formats rather than the OCP
    /// ones cubecl holds: an exponent bias one higher, so a finite code reads as half its OCP
    /// value, and `0x80`, OCP's negative zero, as the one NaN. Only the quantized MMA, which
    /// clears negative zeros and folds the halving into its scales, emits such a form.
    pub fn reads_fnuz(&self) -> bool {
        matches!(self, MfmaForm::Fp8 { .. })
    }

    /// Every tile `generation` lowers: each form at a 16x16 output, one or two instructions deep.
    pub fn configs(generation: AmdMfma) -> Vec<MmaConfig> {
        [(F16, F32), (BF16, F32), (I8, I32)]
            .into_iter()
            .filter_map(|(ab, cd)| Some((ab, cd, Self::of(generation, ab, ab, cd)?)))
            .flat_map(|(ab, cd, form)| {
                TILE_DEPTHS.map(|depth| MmaConfig {
                    a_type: ab,
                    b_type: ab,
                    cd_type: cd,
                    m: INSTRUCTION_M as u32,
                    n: INSTRUCTION_N as u32,
                    k: (depth * form.instruction_k(generation)) as u32,
                })
            })
            .collect()
    }

    /// How deep one instruction is.
    pub fn instruction_k(&self, generation: AmdMfma) -> usize {
        match (self, generation) {
            (MfmaForm::Half { .. }, _) => 16,
            (MfmaForm::Int8, AmdMfma::Cdna2) => 16,
            (MfmaForm::Int8, AmdMfma::Cdna3) | (MfmaForm::Fp8 { .. }, _) => 32,
        }
    }

    /// The intrinsic, which no type overloads.
    fn intrinsic(&self, generation: AmdMfma) -> String {
        let name = match (self, generation) {
            (MfmaForm::Half { ab: BF16 }, _) => "llvm.amdgcn.mfma.f32.16x16x16bf16.1k",
            (MfmaForm::Half { .. }, _) => "llvm.amdgcn.mfma.f32.16x16x16f16",
            (MfmaForm::Int8, AmdMfma::Cdna2) => "llvm.amdgcn.mfma.i32.16x16x16i8",
            (MfmaForm::Int8, AmdMfma::Cdna3) => "llvm.amdgcn.mfma.i32.16x16x32.i8",
            (MfmaForm::Fp8 { a, b }, _) => {
                return format!(
                    "llvm.amdgcn.mfma.f32.16x16x32.{}.{}",
                    fp8_name(*a),
                    fp8_name(*b)
                );
            }
        };
        name.into()
    }
}

/// AMD names `e4m3` `fp8` and `e5m2` `bf8`.
fn fp8_name(elem: ElemType) -> &'static str {
    match elem {
        E4M3 => "fp8",
        _ => "bf8",
    }
}

/// How many instructions a tile takes, where its output is one instruction's and its depth a
/// whole number of them.
pub fn instruction_steps(shape: MatrixShape, instruction_k: usize) -> Option<usize> {
    let MatrixShape { m, n, k } = shape;
    let output = m == INSTRUCTION_M && n == INSTRUCTION_N;
    let depth = k > 0 && k.is_multiple_of(instruction_k);
    (output && depth).then_some(k / instruction_k)
}

/// Emits the instructions for a tile, or `None` where no whole number of them makes its shape.
///
/// A deeper tile is cut into consecutive slices of each lane's fragment. A lane holds its
/// group's run of `k`, so a slice pairs `k` from all four groups the instruction would not, but
/// A and B are sliced alike and the products summed are the same.
pub fn emit_mfma(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    (form, generation): (MfmaForm, AmdMfma),
    shape: MatrixShape,
    (a_val, b_val, c_val): (Value, Value, Value),
) -> Option<Value> {
    let instruction_k = form.instruction_k(generation);
    let steps = instruction_steps(shape, instruction_k)?;
    let slice_width = instruction_k / MFMA_LANE_GROUPS;
    let cd_ty = c_val.get_type(ctx);

    let no_broadcast = insert_i32_const(ctx, rw, NO_BROADCAST);
    let mut acc = c_val;
    for step in 0..steps {
        let a_slice = super::matrix::fragment_slice(ctx, rw, a_val, step, slice_width);
        let b_slice = super::matrix::fragment_slice(ctx, rw, b_val, step, slice_width);
        let a_arg = mfma_operand(ctx, rw, a_slice);
        let b_arg = mfma_operand(ctx, rw, b_slice);
        let args = vec![a_arg, b_arg, acc, no_broadcast, no_broadcast, no_broadcast];
        acc = call_intrinsic(ctx, rw, &form.intrinsic(generation), cd_ty, args);
    }
    Some(acc)
}

/// A fragment slice as the MFMA intrinsics take it: the `bf16` form declares its lanes as `i16`,
/// and the 8-bit forms pack the whole slice into one integer.
fn mfma_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, slice: Value) -> Value {
    let ty = slice.get_type(ctx);
    let (elem, lanes) = {
        let vector = ty.deref(ctx);
        let vector = vector
            .downcast_ref::<LlvmVectorType>()
            .expect("a fragment is held in a vector");
        (vector.elem_type(), vector.num_elements())
    };
    let packed = if elem.deref(ctx).is::<BF16Type>() {
        let halves = IntegerType::get(ctx, 16, Signedness::Signless).into();
        LlvmVectorType::get(ctx, halves, lanes, VectorTypeKind::Fixed).into()
    } else if elem
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|int| int.width() == 8)
    {
        IntegerType::get(ctx, 8 * lanes, Signedness::Signless).into()
    } else {
        return slice;
    };
    bitcast(ctx, rw, slice, packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mfma_accumulates_in_32_bits_from_matching_operands() {
        assert_eq!(
            MfmaForm::of(AmdMfma::Cdna3, F16, F16, F32),
            Some(MfmaForm::Half { ab: F16 })
        );
        assert_eq!(
            MfmaForm::of(AmdMfma::Cdna3, BF16, BF16, F32),
            Some(MfmaForm::Half { ab: BF16 })
        );
        assert_eq!(
            MfmaForm::of(AmdMfma::Cdna3, I8, I8, I32),
            Some(MfmaForm::Int8)
        );
        assert_eq!(MfmaForm::of(AmdMfma::Cdna3, F16, F16, F16), None);
        assert_eq!(MfmaForm::of(AmdMfma::Cdna3, F16, BF16, F32), None);
        let u8 = ElemType::UInt(cubecl_core::ir::UIntKind::U8);
        assert_eq!(
            MfmaForm::of(AmdMfma::Cdna3, u8, I8, I32),
            None,
            "MFMA has no unsigned form"
        );
    }

    #[test]
    fn cdna3_doubled_the_depth_of_its_integer_form() {
        assert_eq!(MfmaForm::Int8.instruction_k(AmdMfma::Cdna2), 16);
        assert_eq!(MfmaForm::Int8.instruction_k(AmdMfma::Cdna3), 32);
        assert_eq!(
            MfmaForm::Int8.intrinsic(AmdMfma::Cdna3),
            "llvm.amdgcn.mfma.i32.16x16x32.i8"
        );
        assert_eq!(
            MfmaForm::Half { ab: BF16 }.intrinsic(AmdMfma::Cdna3),
            "llvm.amdgcn.mfma.f32.16x16x16bf16.1k"
        );
    }

    #[test]
    fn each_form_is_offered_one_and_two_instructions_deep() {
        let configs = MfmaForm::configs(AmdMfma::Cdna3);
        assert_eq!(configs.len(), 3 * 2);
        let depths = |ab| {
            configs
                .iter()
                .filter(|config| config.a_type == ab)
                .map(|config| config.k)
                .collect::<Vec<_>>()
        };
        assert_eq!(depths(F16), [16, 32]);
        assert_eq!(depths(I8), [32, 64]);
        assert!(
            configs
                .iter()
                .all(|config| config.m == 16 && config.n == 16)
        );
    }

    #[test]
    fn a_tile_is_a_whole_number_of_instructions() {
        let shape = |m, n, k| MatrixShape { m, n, k };
        assert_eq!(instruction_steps(shape(16, 16, 32), 16), Some(2));
        assert_eq!(instruction_steps(shape(16, 16, 16), 32), None);
        assert_eq!(instruction_steps(shape(32, 32, 16), 16), None);
        assert_eq!(instruction_steps(shape(16, 16, 0), 16), None);
    }
}
