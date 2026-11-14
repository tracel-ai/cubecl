//! CDNA's MFMA instructions, the matrix cores of MI200 and MI300.
//!
//! Every form this lowering emits writes a 16x16 tile over a 64-wide wavefront, laid out as
//! AMD's matrix instruction calculator gives it: lane `l` sits in row group `l / 16` at column
//! `l % 16`. Of A it holds row `l % 16` and of B column `l % 16`, each a run of
//! `instruction_k / 4` consecutive `k` starting at `(l / 16) * instruction_k / 4`. Of the
//! accumulator it holds column `l % 16`, rows `4 * (l / 16)` to `4 * (l / 16) + 3`.

use super::fragment::{
    BF16, BYTE_BITS, E4M3, E5M2, F16, F32, HALF_BITS, I8, I32, INSTRUCTION_M, INSTRUCTION_N,
    TILE_DEPTHS, fp8_name, fragment_slice, instruction_steps,
};
use crate::{prelude::*, shared::plane::bitcast};
use cubecl_core::ir::{
    ElemType,
    amd::{AmdMfma, MFMA_LANE_GROUPS},
    features::MmaConfig,
    types::MatrixShape,
};

/// Accumulator elements each lane holds: a 16x16 output over 64 lanes.
pub const ACCUMULATOR_REGISTERS: usize = 4;

/// The `cbsz`, `abid` and `blgp` modifiers, which broadcast and permute blocks of A and B. The
/// 16x16 single-block forms take none.
const NO_BROADCAST: i32 = 0;

/// The depth of the 16-bit float forms and of CDNA2's integer one.
const SHALLOW_K: usize = 16;

/// The depth of CDNA3's 8-bit forms.
const DEEP_K: usize = 32;

/// What one product of an FNUZ form reads its operands at: half each, so a quarter.
const FNUZ_PRODUCT_FACTOR: f32 = 4.0;

/// The MFMA instruction a tile's element types select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MfmaForm {
    /// `f16` or `bf16` operands accumulating in `f32`, 16 deep.
    Half { ab: ElemType },
    /// Signed 8-bit operands accumulating in `i32`: 16 deep on CDNA2, 32 deep on CDNA3.
    Int8,
    /// 8-bit float operands, each side its own format, accumulating in `f32`, 32 deep. CDNA3
    /// only, and it reads its bytes as FNUZ ([`CodeReading::Fnuz`]).
    Fp8 { a: ElemType, b: ElemType },
}

/// How a matrix instruction reads the bytes of its 8-bit float operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeReading {
    /// As the OCP formats cubecl holds, or the form reads no 8-bit floats.
    Ocp,
    /// As AMD's FNUZ formats: an exponent bias one higher, so a finite code reads as half its
    /// OCP value, and `0x80`, OCP's negative zero, as the one NaN. Only the quantized MMA, which
    /// clears negative zeros and folds the halving into its scales, emits such a form.
    Fnuz,
}

impl CodeReading {
    /// What one product reads at, against the OCP values of its operands, where it differs.
    pub fn product_factor(self) -> Option<f32> {
        match self {
            CodeReading::Ocp => None,
            CodeReading::Fnuz => Some(FNUZ_PRODUCT_FACTOR),
        }
    }
}

/// How a fragment slice is handed to an MFMA intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperandCarriage {
    /// In the vector it is held in.
    AsHeld,
    /// Its `bf16` lanes declared as `i16`.
    HalvesAsIntegers,
    /// Its 8-bit lanes packed into one integer.
    PackedBytes,
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

    /// How the form reads its 8-bit float bytes.
    pub fn code_reading(&self) -> CodeReading {
        match self {
            MfmaForm::Fp8 { .. } => CodeReading::Fnuz,
            MfmaForm::Half { .. } | MfmaForm::Int8 => CodeReading::Ocp,
        }
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
            (MfmaForm::Half { .. }, _) | (MfmaForm::Int8, AmdMfma::Cdna2) => SHALLOW_K,
            (MfmaForm::Int8, AmdMfma::Cdna3) | (MfmaForm::Fp8 { .. }, _) => DEEP_K,
        }
    }

    /// The intrinsic, which no type overloads.
    fn intrinsic(&self, generation: AmdMfma) -> String {
        match (self, generation) {
            (MfmaForm::Half { ab: BF16 }, _) => "llvm.amdgcn.mfma.f32.16x16x16bf16.1k".into(),
            (MfmaForm::Half { .. }, _) => "llvm.amdgcn.mfma.f32.16x16x16f16".into(),
            (MfmaForm::Int8, AmdMfma::Cdna2) => "llvm.amdgcn.mfma.i32.16x16x16i8".into(),
            (MfmaForm::Int8, AmdMfma::Cdna3) => "llvm.amdgcn.mfma.i32.16x16x32.i8".into(),
            (MfmaForm::Fp8 { a, b }, _) => format!(
                "llvm.amdgcn.mfma.f32.16x16x32.{}.{}",
                fp8_name(*a),
                fp8_name(*b)
            ),
        }
    }

    /// How the form takes its operands: the `bf16` form declares its lanes as `i16`, the 8-bit
    /// forms pack a slice into one integer.
    fn carriage(&self) -> OperandCarriage {
        match self {
            MfmaForm::Half { ab: BF16 } => OperandCarriage::HalvesAsIntegers,
            MfmaForm::Half { .. } => OperandCarriage::AsHeld,
            MfmaForm::Int8 | MfmaForm::Fp8 { .. } => OperandCarriage::PackedBytes,
        }
    }
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
    let carriage = form.carriage();

    let no_broadcast = insert_i32_const(ctx, rw, NO_BROADCAST);
    let mut acc = c_val;
    for step in 0..steps {
        let a_slice = fragment_slice(ctx, rw, a_val, step, slice_width);
        let b_slice = fragment_slice(ctx, rw, b_val, step, slice_width);
        let a_arg = mfma_operand(ctx, rw, a_slice, carriage);
        let b_arg = mfma_operand(ctx, rw, b_slice, carriage);
        let args = vec![a_arg, b_arg, acc, no_broadcast, no_broadcast, no_broadcast];
        acc = call_intrinsic(ctx, rw, &form.intrinsic(generation), cd_ty, args);
    }
    Some(acc)
}

/// A fragment slice as the MFMA intrinsics take it, carried as `carriage` says.
fn mfma_operand(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    slice: Value,
    carriage: OperandCarriage,
) -> Value {
    let lanes = slice
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .expect("a fragment is held in a vector")
        .num_elements();
    let carried = match carriage {
        OperandCarriage::AsHeld => return slice,
        OperandCarriage::HalvesAsIntegers => {
            let halves = IntegerType::get(ctx, HALF_BITS, Signedness::Signless).into();
            LlvmVectorType::get(ctx, halves, lanes, VectorTypeKind::Fixed).into()
        }
        OperandCarriage::PackedBytes => {
            IntegerType::get(ctx, BYTE_BITS * lanes, Signedness::Signless).into()
        }
    };
    bitcast(ctx, rw, slice, carried)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amdgpu::fragment::U8;

    #[test]
    fn mfma_accumulates_in_32_bits_from_matching_operands() {
        let cdna3 = AmdMfma::Cdna3;
        assert_eq!(
            MfmaForm::of(cdna3, F16, F16, F32),
            Some(MfmaForm::Half { ab: F16 })
        );
        assert_eq!(
            MfmaForm::of(cdna3, BF16, BF16, F32),
            Some(MfmaForm::Half { ab: BF16 })
        );
        assert_eq!(MfmaForm::of(cdna3, I8, I8, I32), Some(MfmaForm::Int8));
        assert_eq!(MfmaForm::of(cdna3, F16, F16, F16), None);
        assert_eq!(MfmaForm::of(cdna3, F16, BF16, F32), None);
        assert_eq!(
            MfmaForm::of(cdna3, U8, I8, I32),
            None,
            "MFMA has no unsigned form"
        );
    }

    /// Only CDNA3 has 8-bit float forms, and they read their bytes as FNUZ.
    #[test]
    fn cdna3_reads_its_8_bit_floats_as_fnuz() {
        let fp8 = MfmaForm::of(AmdMfma::Cdna3, E4M3, E5M2, F32);
        assert_eq!(fp8, Some(MfmaForm::Fp8 { a: E4M3, b: E5M2 }));
        assert_eq!(fp8.map(|form| form.code_reading()), Some(CodeReading::Fnuz));
        assert_eq!(MfmaForm::of(AmdMfma::Cdna2, E4M3, E4M3, F32), None);
        assert_eq!(MfmaForm::Int8.code_reading(), CodeReading::Ocp);
    }

    #[test]
    fn cdna3_doubled_the_depth_of_its_integer_form() {
        assert_eq!(MfmaForm::Int8.instruction_k(AmdMfma::Cdna2), SHALLOW_K);
        assert_eq!(MfmaForm::Int8.instruction_k(AmdMfma::Cdna3), DEEP_K);
        assert_eq!(
            MfmaForm::Int8.intrinsic(AmdMfma::Cdna3),
            "llvm.amdgcn.mfma.i32.16x16x32.i8"
        );
        assert_eq!(
            MfmaForm::Half { ab: BF16 }.intrinsic(AmdMfma::Cdna3),
            "llvm.amdgcn.mfma.f32.16x16x16bf16.1k"
        );
        let fp8 = MfmaForm::Fp8 { a: E5M2, b: E4M3 };
        assert_eq!(
            fp8.intrinsic(AmdMfma::Cdna3),
            "llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8"
        );
    }

    #[test]
    fn each_form_is_offered_one_and_two_instructions_deep() {
        let configs = MfmaForm::configs(AmdMfma::Cdna3);
        let forms = 3;
        assert_eq!(configs.len(), forms * TILE_DEPTHS.len());
        let depths = |ab| {
            configs
                .iter()
                .filter(|config| config.a_type == ab)
                .map(|config| config.k as usize)
                .collect::<Vec<_>>()
        };
        assert_eq!(depths(F16), [SHALLOW_K, 2 * SHALLOW_K]);
        assert_eq!(depths(I8), [DEEP_K, 2 * DEEP_K]);
        assert!(
            configs
                .iter()
                .all(|config| config.m == 16 && config.n == 16)
        );
    }
}
