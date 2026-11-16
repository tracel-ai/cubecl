//! The quantized MMA on AMDGPU: the tile's codes multiplied by the part's plain MMA into a zero
//! sum, which is then scaled into the accumulator.
//!
//! A tile lies within one block of scales, so one product of the codes takes one pair of scales
//! per accumulator element. Integer codes sum in `i32` and are converted; 8-bit floats sum in
//! `f32`. MI300's 8-bit float MFMA reads its bytes as FNUZ: each finite OCP code reads as half
//! its value, so the product as a quarter, which the scale makes up for, and OCP's negative zero
//! reads as NaN, so it is cleared to a positive zero first. OCP NaN and infinity codes read as
//! large finite values there, which a saturating quantizer never stores.

use super::*;
use crate::amdgpu::fragment::{F32, I32, INSTRUCTION_M, INSTRUCTION_N, TILE_DEPTHS, WORD_BITS};
use crate::shared::matrix::{registers_as_vector, registers_value};
use cubecl_core::ir::{dialect::matrix::MmaQuantizedOp, features::QuantizedMmaConfig};
use cubecl_core::quant::scheme::{QuantValue, ScaleDtype};
use pliron::{attribute::boxed_attr_cast, builtin::attributes::FPSingleAttr};

#[derive(Debug, Error)]
#[error(
    "no quantized MMA lowering for a {}x{}x{} tile under blocks of {block_k}: a tile is \
     {INSTRUCTION_M}x{INSTRUCTION_N}, a multiple of {instruction_k} deep, and lies within one \
     block",
    shape.m, shape.n, shape.k
)]
pub struct QuantizedShapeUnsupported {
    pub shape: MatrixShape,
    pub block_k: usize,
    pub instruction_k: usize,
}

/// The byte masks clearing a negative zero four bytes at a time.
const LOW_SEVEN_BITS: i32 = 0x7F7F_7F7F;
const HIGH_BITS: i32 = 0x8080_8080_u32 as i32;

/// The 8-bit codes one 32-bit word holds.
const CODES_PER_WORD: u32 = 4;

/// The sums a product of the codes starts from.
const INTEGER_ZERO: i32 = 0;
const FLOAT_ZERO: f32 = 0.0;

/// What the codes' product sums in before it is scaled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sum {
    /// Integer codes, summed in `i32`.
    Integer,
    /// 8-bit float codes, summed in `f32`.
    Float,
}

impl Sum {
    /// What a product of `codes` sums in.
    fn of(codes: ElemType) -> Self {
        match codes {
            ElemType::Int(_) => Sum::Integer,
            _ => Sum::Float,
        }
    }

    /// The element the sum is held in.
    fn elem(self) -> ElemType {
        match self {
            Sum::Integer => I32,
            Sum::Float => F32,
        }
    }

    /// A zero sum, one lane of it.
    fn zero(self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        match self {
            Sum::Integer => insert_i32_const(ctx, rw, INTEGER_ZERO),
            Sum::Float => insert_f32_const(ctx, rw, FLOAT_ZERO),
        }
    }

    /// `sum` as the `f32` vector `f32_ty` the accumulator is.
    fn to_f32(
        self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        sum: Value,
        f32_ty: TypeHandle,
    ) -> Value {
        match self {
            Sum::Integer => sitofp(ctx, rw, sum, f32_ty),
            Sum::Float => sum,
        }
    }
}

pub fn mma_quantized(
    op: &MmaQuantizedOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let (a, b, c, d) = (
        op.registers_a(ctx),
        op.registers_b(ctx),
        op.registers_c(ctx),
        op.registers_d(ctx),
    );
    let (scales_a, scales_b) = (op.scales_a(ctx), op.scales_b(ctx));
    let block_k = op.block_k(ctx).0;
    let shape = *op.shape(ctx).clone();

    let (a_ty, a_elem) = registers_as_vector(ctx, operands_info, a);
    let (b_ty, b_elem) = registers_as_vector(ctx, operands_info, b);
    let (cd_ty, _) = registers_as_vector(ctx, operands_info, c);
    let (scales_ty, _) = registers_as_vector(ctx, operands_info, scales_a);

    let sum = Sum::of(elem_type(ctx, a_elem));
    let sum_elem = sum.elem().to_type(ctx);
    let form = match MatrixForm::any(ctx, a_elem, b_elem, sum_elem) {
        Ok(form) => form,
        Err(err) => return input_err!(op.loc(ctx), err),
    };
    let unsupported = QuantizedShapeUnsupported {
        shape,
        block_k,
        instruction_k: form.instruction_k(),
    };
    if !shape.k.is_multiple_of(unsupported.instruction_k) || !block_k.is_multiple_of(shape.k) {
        return input_err!(op.loc(ctx), unsupported);
    }

    let a_val = registers_value(ctx, rw, a, a_ty);
    let b_val = registers_value(ctx, rw, b, b_ty);
    let c_val = registers_value(ctx, rw, c, cd_ty);
    let scales_a = registers_value(ctx, rw, scales_a, scales_ty);
    let scales_b = registers_value(ctx, rw, scales_b, scales_ty);
    let (a_val, b_val) = match form.code_reading() {
        CodeReading::Ocp => (a_val, b_val),
        CodeReading::Fnuz => (
            clear_negative_zeros(ctx, rw, a_val),
            clear_negative_zeros(ctx, rw, b_val),
        ),
    };

    let sums = vector_lanes(ctx, cd_ty);
    let sum_scalar = cube_type_to_llvm(ctx, sum_elem);
    let sum_ty = LlvmVectorType::get(ctx, sum_scalar, sums, VectorTypeKind::Fixed).into();
    let zero = sum.zero(ctx, rw);
    let zero = insert_splat(ctx, rw, sum_ty, zero, sums as usize);
    let Some(product) = form.emit(ctx, rw, shape, (a_val, b_val, zero), sum_ty) else {
        return input_err!(op.loc(ctx), unsupported);
    };
    let product = sum.to_f32(ctx, rw, product, cd_ty);

    let scale = fmul(ctx, rw, scales_a, scales_b);
    let scale = match form.code_reading().product_factor() {
        None => scale,
        Some(factor) => scaled_by(ctx, rw, scale, factor),
    };
    let fma = format!("llvm.fma.{}", llvm_mangled_ty(ctx, cd_ty));
    let result = call_intrinsic(ctx, rw, &fma, cd_ty, vec![product, scale, c_val]);
    store_fragment(ctx, rw, d, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// The lanes of a fragment vector.
fn vector_lanes(ctx: &Context, ty: TypeHandle) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .expect("a fragment is held in a vector")
        .num_elements()
}

fn insert_f32_const(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: f32) -> Value {
    let attr = boxed_attr_cast(FPSingleAttr::from(value).into()).expect("an f32 constant");
    let op = llvm::ConstantOp::new(ctx, attr);
    insert(ctx, rw, &op)
}

fn fmul(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op = llvm::FMulOp::new_with_fast_math_flags(ctx, lhs, rhs, FastmathFlagsAttr::default());
    insert(ctx, rw, &op)
}

fn sitofp(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::SIToFPOp::new(ctx, value, ty);
    insert(ctx, rw, &op)
}

fn and(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op = llvm::AndOp::new(ctx, lhs, rhs);
    insert(ctx, rw, &op)
}

fn or(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op = llvm::OrOp::new(ctx, lhs, rhs);
    insert(ctx, rw, &op)
}

/// The `f32` lanes of `values`, each multiplied by `factor`.
fn scaled_by(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    values: Value,
    factor: f32,
) -> Value {
    let ty = values.get_type(ctx);
    let lanes = vector_lanes(ctx, ty);
    let factor = insert_f32_const(ctx, rw, factor);
    let factor = insert_splat(ctx, rw, ty, factor, lanes as usize);
    fmul(ctx, rw, values, factor)
}

/// `codes`, bytes held in a vector, with every `0x80`, OCP's negative zero and FNUZ's NaN,
/// cleared to `0x00`, four bytes to an `i32`: a byte keeps its sign bit only where its low seven
/// bits are not all zero. Mirrors [`clear_negative_zero_word`].
fn clear_negative_zeros(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    codes: Value,
) -> Value {
    let bytes_ty = codes.get_type(ctx);
    let words = vector_lanes(ctx, bytes_ty) / CODES_PER_WORD;
    let word_ty = IntegerType::get(ctx, WORD_BITS, Signedness::Signless).into();
    let words_ty = LlvmVectorType::get(ctx, word_ty, words, VectorTypeKind::Fixed).into();
    let mut splat = |value: i32| {
        let scalar = insert_i32_const(ctx, rw, value);
        insert_splat(ctx, rw, words_ty, scalar, words as usize)
    };
    let (low_bits, high_bits) = (splat(LOW_SEVEN_BITS), splat(HIGH_BITS));

    let word = bitcast(ctx, rw, codes, words_ty);
    let low = and(ctx, rw, word, low_bits);
    let carried = add(ctx, rw, low, low_bits);
    let nonzero = and(ctx, rw, carried, high_bits);
    let keep = or(ctx, rw, nonzero, low_bits);
    let cleared = and(ctx, rw, word, keep);
    bitcast(ctx, rw, cleared, bytes_ty)
}

/// [`clear_negative_zeros`] on one word, as the lowering computes it.
#[cfg(test)]
fn clear_negative_zero_word(word: u32) -> u32 {
    let (low_bits, high_bits) = (LOW_SEVEN_BITS as u32, HIGH_BITS as u32);
    let low = word & low_bits;
    let nonzero = low.wrapping_add(low_bits) & high_bits;
    word & (nonzero | low_bits)
}

/// The value pairs whose codes an 8-bit float form multiplies.
const FP8_PAIRS: [(QuantValue, QuantValue); 4] = [
    (QuantValue::E4M3, QuantValue::E4M3),
    (QuantValue::E4M3, QuantValue::E5M2),
    (QuantValue::E5M2, QuantValue::E4M3),
    (QuantValue::E5M2, QuantValue::E5M2),
];

/// The value pairs whose codes an 8-bit integer form multiplies: both are held in `i8`, so each
/// pair lowers alike, and each is offered so a scheme of either finds its own.
const INT8_PAIRS: [(QuantValue, QuantValue); 4] = [
    (QuantValue::Q8S, QuantValue::Q8S),
    (QuantValue::Q8S, QuantValue::Q8F),
    (QuantValue::Q8F, QuantValue::Q8S),
    (QuantValue::Q8F, QuantValue::Q8F),
];

/// Every scale type a block may be stored in: each is decoded to `f32` before the product.
const SCALES: [ScaleDtype; 5] = [
    ScaleDtype::F32,
    ScaleDtype::F16,
    ScaleDtype::BF16,
    ScaleDtype::UE8M0,
    ScaleDtype::UE4M3,
];

/// The block sizes along `k` offered.
const BLOCK_KS: [usize; 3] = [32, 64, 128];

/// Every quantized MMA `core` lowers: each value pair it has a form for, under every scale type
/// and block, at a 16x16 output one or two instructions deep and within one block. The form is
/// the one [`mma_quantized`] lowers to, chosen the same way.
pub fn quantized_configs(core: AmdMatrixCore) -> Vec<QuantizedMmaConfig> {
    let instruction_k = |a: QuantValue, b: QuantValue| {
        let (a, b) = (ElemType::from_quant_value(a), ElemType::from_quant_value(b));
        let form = MatrixForm::of(core, a, b, Sum::of(a).elem())?;
        Some(form.instruction_k())
    };
    FP8_PAIRS
        .into_iter()
        .chain(INT8_PAIRS)
        .filter_map(|(a, b)| Some((a, b, instruction_k(a, b)?)))
        .flat_map(|(a, b, instruction_k)| {
            SCALES.into_iter().flat_map(move |scales| {
                TILE_DEPTHS.into_iter().flat_map(move |depth| {
                    let k = depth * instruction_k;
                    BLOCK_KS
                        .into_iter()
                        .filter(move |block_k| block_k.is_multiple_of(k))
                        .map(move |block_k| QuantizedMmaConfig {
                            a,
                            b,
                            scales,
                            block_k: block_k as u32,
                            m: INSTRUCTION_M as u32,
                            n: INSTRUCTION_N as u32,
                            k: k as u32,
                        })
                })
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::ir::amd::{AmdMfma, AmdWmma};

    /// FNUZ's one NaN, where OCP has its negative zero.
    const FNUZ_NAN: u8 = 0x80;
    const SIGN_BIT: u8 = 0x80;
    const MAGNITUDE_BITS: u8 = 0x7F;
    const CODE_BITS: u32 = 7;

    /// An FNUZ byte decoded: bias one higher than OCP's, no infinity, `0x80` the one NaN. A
    /// subnormal and a normal code read alike once the implicit leading one is the exponent's
    /// own: zero below the smallest normal exponent, one from it on.
    fn fnuz_value(code: u8, exponent_bits: u32) -> f64 {
        if code == FNUZ_NAN {
            return f64::NAN;
        }
        let mantissa_bits = CODE_BITS - exponent_bits;
        let bias: i32 = 1 << (exponent_bits - 1);
        let sign = 1.0 - 2.0 * f64::from((code & SIGN_BIT) >> CODE_BITS);
        let magnitude = u32::from(code & MAGNITUDE_BITS);
        let exponent = (magnitude >> mantissa_bits) as i32;
        let mantissa = f64::from(magnitude & ((1 << mantissa_bits) - 1));
        let leading_one = f64::from(exponent.min(1));
        let fraction = mantissa / f64::from(1u32 << mantissa_bits);
        sign * (leading_one + fraction) * 2f64.powi(exponent.max(1) - bias)
    }

    /// Every finite OCP code but the negative zero reads as half its value once its word is
    /// cleared, in both formats, and the negative zero as a zero: the product's factor of four
    /// and the clearing together read every finite code exactly.
    #[test]
    fn a_cleared_ocp_code_reads_as_half_its_value_in_fnuz() {
        use cubecl_core::{e4m3, e5m2};
        const NEIGHBOUR: u8 = 0x01;
        const NEGATIVE_ZERO: u8 = 0x80;
        for code in 0..=u8::MAX {
            let word = u32::from_le_bytes([code, NEIGHBOUR, NEGATIVE_ZERO, code]);
            let cleared = clear_negative_zero_word(word).to_le_bytes();
            assert_eq!(cleared[1], NEIGHBOUR, "a code beside it is untouched");
            assert_eq!(
                cleared[2], 0,
                "{code:#04x}: a negative zero beside it is cleared"
            );
            assert_eq!(cleared[0], cleared[3]);
            let read = cleared[0];

            let formats = [
                ("e4m3", e4m3::from_bits(code).to_f64(), 4),
                ("e5m2", e5m2::from_bits(code).to_f64(), 5),
            ];
            for (format, ocp, exponent_bits) in formats {
                if !ocp.is_finite() {
                    continue;
                }
                let fnuz = fnuz_value(read, exponent_bits);
                assert_eq!(fnuz * 2.0, ocp, "{format} {code:#04x}");
            }
        }
    }

    /// MI300 offers 8-bit floats and integers, MI200 integers only, RDNA4 both and RDNA3
    /// integers, every tile lying within one block.
    #[test]
    fn each_core_offers_the_codes_it_multiplies() {
        let offers =
            |configs: &[QuantizedMmaConfig], value| configs.iter().any(|config| config.a == value);
        for (core, has_fp8) in [
            (AmdMatrixCore::Mfma(AmdMfma::Cdna3), true),
            (AmdMatrixCore::Mfma(AmdMfma::Cdna2), false),
            (AmdMatrixCore::Wmma(AmdWmma::Rdna4), true),
            (AmdMatrixCore::Wmma(AmdWmma::Rdna3), false),
        ] {
            let configs = quantized_configs(core);
            assert_eq!(offers(&configs, QuantValue::E4M3), has_fp8, "{core:?}");
            assert!(offers(&configs, QuantValue::Q8S), "{core:?}");
            assert!(
                configs
                    .iter()
                    .all(|config| config.block_k.is_multiple_of(config.k)),
                "{core:?}"
            );
        }
        let cdna3 = quantized_configs(AmdMatrixCore::Mfma(AmdMfma::Cdna3));
        assert!(cdna3.contains(&QuantizedMmaConfig {
            a: QuantValue::E4M3,
            b: QuantValue::E5M2,
            scales: ScaleDtype::UE8M0,
            block_k: 32,
            m: 16,
            n: 16,
            k: 32,
        }));
        let fp8_depth = 32;
        assert!(
            cdna3.iter().all(|config| config.k >= fp8_depth),
            "MI300's forms are 32 deep"
        );
    }
}
