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
use crate::shared::matrix::{elem_type, registers_as_vector, registers_value};
use cubecl_core::ir::{dialect::matrix::MmaQuantizedOp, features::QuantizedMmaConfig};
use cubecl_core::quant::scheme::{QuantValue, ScaleDtype};
use pliron::{attribute::boxed_attr_cast, builtin::attributes::FPSingleAttr};

#[derive(Debug, Error)]
#[error(
    "no quantized MMA lowering for a {m}x{n}x{k} tile under blocks of {block_k}: a tile is \
     {INSTRUCTION_M}x{INSTRUCTION_N}, a multiple of {instruction_k} deep, and lies within one \
     block",
    m = .0.m, n = .0.n, k = .0.k, block_k = .1, instruction_k = .2
)]
pub struct QuantizedShapeUnsupported(MatrixShape, usize, usize);

/// What one FNUZ product reads its operands at: half each, so a quarter of the product.
const FNUZ_PRODUCT_FACTOR: f32 = 4.0;

/// The byte masks clearing a negative zero four bytes at a time.
const LOW_SEVEN_BITS: i32 = 0x7F7F_7F7F;
const HIGH_BITS: i32 = 0x8080_8080_u32 as i32;

pub(crate) fn mma_quantized(
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

    let integer = matches!(elem_type(ctx, a_elem), ElemType::Int(_));
    let sum_elem = match integer {
        true => ElemType::Int(IntKind::I32),
        false => F32,
    }
    .to_type(ctx);
    let form = match MatrixForm::any(ctx, a_elem, b_elem, sum_elem) {
        Ok(form) => form,
        Err(err) => return input_err!(op.loc(ctx), err),
    };
    let instruction_k = form.instruction_k();
    if !shape.k.is_multiple_of(instruction_k) || !block_k.is_multiple_of(shape.k) {
        return input_err!(
            op.loc(ctx),
            QuantizedShapeUnsupported(shape, block_k, instruction_k)
        );
    }

    let a_val = registers_value(ctx, rw, a, a_ty);
    let b_val = registers_value(ctx, rw, b, b_ty);
    let c_val = registers_value(ctx, rw, c, cd_ty);
    let scales_a = registers_value(ctx, rw, scales_a, scales_ty);
    let scales_b = registers_value(ctx, rw, scales_b, scales_ty);
    let (a_val, b_val) = match form.reads_fnuz() {
        true => (
            clear_negative_zeros(ctx, rw, a_val),
            clear_negative_zeros(ctx, rw, b_val),
        ),
        false => (a_val, b_val),
    };

    let sums = vector_lanes(ctx, cd_ty);
    let sum_scalar = cube_type_to_llvm(ctx, sum_elem);
    let sum_ty = LlvmVectorType::get(ctx, sum_scalar, sums, VectorTypeKind::Fixed).into();
    let zero = match integer {
        true => insert_i32_const(ctx, rw, 0),
        false => insert_f32_const(ctx, rw, 0.0),
    };
    let zero = insert_splat(ctx, rw, sum_ty, zero, sums as usize);
    let Some(product) = form.emit(ctx, rw, shape, (a_val, b_val, zero), sum_ty) else {
        return input_err!(
            op.loc(ctx),
            QuantizedShapeUnsupported(shape, block_k, instruction_k)
        );
    };
    let product = match integer {
        true => {
            let op = llvm::SIToFPOp::new(ctx, product, cd_ty);
            insert(ctx, rw, &op)
        }
        false => product,
    };

    let mut scale = fmul(ctx, rw, scales_a, scales_b);
    if form.reads_fnuz() {
        let factor = insert_f32_const(ctx, rw, FNUZ_PRODUCT_FACTOR);
        let factor = insert_splat(ctx, rw, cd_ty, factor, sums as usize);
        scale = fmul(ctx, rw, scale, factor);
    }
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

/// `codes`, bytes held in a vector, with every `0x80`, OCP's negative zero and FNUZ's NaN,
/// cleared to `0x00`, four bytes to an `i32`: a byte keeps its sign bit only where its low seven
/// bits are not all zero. Mirrors [`clear_negative_zero_word`].
fn clear_negative_zeros(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    codes: Value,
) -> Value {
    let bytes_ty = codes.get_type(ctx);
    let words = vector_lanes(ctx, bytes_ty) / 4;
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let words_ty = LlvmVectorType::get(ctx, i32_ty, words, VectorTypeKind::Fixed).into();
    let mut splat = |value: i32| {
        let scalar = insert_i32_const(ctx, rw, value);
        insert_splat(ctx, rw, words_ty, scalar, words as usize)
    };
    let (low_bits, high_bits) = (splat(LOW_SEVEN_BITS), splat(HIGH_BITS));

    let word = bitcast(ctx, rw, codes, words_ty);
    let and = |ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs, rhs| {
        let op = llvm::AndOp::new(ctx, lhs, rhs);
        insert(ctx, rw, &op)
    };
    let low = and(ctx, rw, word, low_bits);
    let carried = add(ctx, rw, low, low_bits);
    let nonzero = and(ctx, rw, carried, high_bits);
    let keep = {
        let op = llvm::OrOp::new(ctx, nonzero, low_bits);
        insert(ctx, rw, &op)
    };
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

/// The value pairs whose codes an 8-bit integer form multiplies: both are held in `i8`.
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

/// How many instructions deep the advertised tiles are: one, or two in a row.
const TILE_DEPTHS: [usize; 2] = [1, 2];

/// Every quantized MMA `core` lowers: each value pair it has a form for, under every scale type
/// and block, at a 16x16 output one or two instructions deep and within one block.
pub fn quantized_configs(core: AmdMatrixCore) -> Vec<QuantizedMmaConfig> {
    let instruction_k = |a: QuantValue, b: QuantValue| {
        let (a, b) = (ElemType::from_quant_value(a), ElemType::from_quant_value(b));
        let sum = match a {
            ElemType::Int(_) => ElemType::Int(IntKind::I32),
            _ => F32,
        };
        match core {
            AmdMatrixCore::Wmma(generation) => {
                WmmaForm::of(generation, a, b, sum).map(|_| INSTRUCTION_K)
            }
            AmdMatrixCore::Mfma(generation) => {
                MfmaForm::of(generation, a, b, sum).map(|form| form.instruction_k(generation))
            }
        }
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

    /// An FNUZ byte decoded: bias one higher than OCP's, no infinity, `0x80` the one NaN.
    fn fnuz_value(code: u8, exponent_bits: u32) -> f64 {
        let mantissa_bits = 7 - exponent_bits;
        let bias: i32 = 1 << (exponent_bits - 1);
        if code == 0x80 {
            return f64::NAN;
        }
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = ((code as u32 >> mantissa_bits) & ((1 << exponent_bits) - 1)) as i32;
        let mantissa = (code as u32 & ((1 << mantissa_bits) - 1)) as f64;
        let fraction = mantissa / (1 << mantissa_bits) as f64;
        sign * match exponent {
            0 => fraction * 2f64.powi(1 - bias),
            _ => (1.0 + fraction) * 2f64.powi(exponent - bias),
        }
    }

    /// Every finite OCP code but the negative zero reads as half its value once its word is
    /// cleared, in both formats, and the negative zero as a zero: the product's factor of four
    /// and the clearing together read every finite code exactly.
    #[test]
    fn a_cleared_ocp_code_reads_as_half_its_value_in_fnuz() {
        use cubecl_core::{e4m3, e5m2};
        for code in 0..=u8::MAX {
            let word = u32::from_le_bytes([code, 0x01, 0x80, code]);
            let cleared = clear_negative_zero_word(word).to_le_bytes();
            assert_eq!(cleared[1], 0x01, "a code beside it is untouched");
            assert_eq!(
                cleared[2], 0x00,
                "{code:#04x}: a negative zero beside it is cleared"
            );
            assert_eq!(cleared[0], cleared[3]);
            let read = cleared[0];

            let ocp = e4m3::from_bits(code).to_f64();
            if ocp.is_finite() {
                assert_eq!(fnuz_value(read, 4) * 2.0, ocp, "e4m3 {code:#04x}");
            }
            let ocp = e5m2::from_bits(code).to_f64();
            if ocp.is_finite() {
                assert_eq!(fnuz_value(read, 5) * 2.0, ocp, "e5m2 {code:#04x}");
            }
        }
    }

    /// MI300 offers 8-bit floats and integers, MI200 integers only, RDNA4 both and RDNA3
    /// integers, every tile lying within one block.
    #[test]
    fn each_core_offers_the_codes_it_multiplies() {
        let fp8 = |configs: &[QuantizedMmaConfig]| {
            configs.iter().any(|config| config.a == QuantValue::E4M3)
        };
        let int8 = |configs: &[QuantizedMmaConfig]| {
            configs.iter().any(|config| config.a == QuantValue::Q8S)
        };
        for (core, has_fp8) in [
            (AmdMatrixCore::Mfma(AmdMfma::Cdna3), true),
            (AmdMatrixCore::Mfma(AmdMfma::Cdna2), false),
            (AmdMatrixCore::Wmma(AmdWmma::Rdna4), true),
            (AmdMatrixCore::Wmma(AmdWmma::Rdna3), false),
        ] {
            let configs = quantized_configs(core);
            assert_eq!(fp8(&configs), has_fp8, "{core:?}");
            assert!(int8(&configs), "{core:?}");
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
        assert!(
            cdna3.iter().all(|config| config.k >= 32),
            "MI300's forms are 32 deep"
        );
    }
}
