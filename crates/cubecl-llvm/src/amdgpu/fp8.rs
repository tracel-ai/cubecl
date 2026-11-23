//! RDNA4's own fp8 conversions, an instruction a value decoded and one a pair encoded, where the
//! software codec spends a few dozen integer operations a value.

use crate::{
    prelude::*,
    shared::{
        native_decode::NativeDecode,
        plane::bitcast,
        to_llvm::{
            general::convert_float,
            vector::{extract_lane, insert_vector_construct},
        },
    },
};
use cubecl_core::ir::{
    amd::{AmdMatrixCore, AmdWmma},
    dialect::general::CastOp,
    types::Fp8Format,
};

/// Codes a conversion's 32-bit source holds, any one of which it decodes.
const CODES_PER_WORD: usize = 4;

/// The fp8 formats a part with `matrix` converts itself. RDNA4 reads both as OCP states them;
/// CDNA3's conversions are of the FNUZ formats, so its casts stay the software codec's.
pub(crate) fn native_decodes(matrix: Option<AmdMatrixCore>) -> &'static [NativeDecode] {
    match matrix {
        Some(AmdMatrixCore::Wmma(AmdWmma::Rdna4)) => &[NativeDecode::E4M3, NativeDecode::E5M2],
        Some(AmdMatrixCore::Wmma(AmdWmma::Rdna3) | AmdMatrixCore::Mfma(_)) | None => &[],
    }
}

/// An fp8 format as RDNA4's conversions name it and bound it.
#[derive(Clone, Copy, Debug)]
struct Fp8Conversion {
    name: &'static str,
    largest: f32,
}

impl Fp8Conversion {
    fn of(format: NativeDecode) -> Option<Self> {
        match format {
            NativeDecode::E4M3 => Some(Self {
                name: "fp8",
                largest: Fp8Format::E4M3.max_value(),
            }),
            NativeDecode::E5M2 => Some(Self {
                name: "bf8",
                largest: Fp8Format::E5M2.max_value(),
            }),
            NativeDecode::E2M1x2 => None,
        }
    }
}

/// `cast`, of operands of `input_ty`, where it is an fp8 one the pipeline left for the target.
pub(crate) fn cast(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    cast: &CastOp,
    input_ty: TypeHandle,
) -> Option<Value> {
    let result_ty = cast.result_type(ctx);
    let decoded = NativeDecode::decoded_by(ctx, input_ty, result_ty).and_then(Fp8Conversion::of);
    let encoded = NativeDecode::encoded_by(ctx, input_ty, result_ty).and_then(Fp8Conversion::of);
    let input = cast.input(ctx);
    let ty = cube_type_to_llvm(ctx, result_ty);
    match (decoded, encoded) {
        (Some(format), _) => Some(decode(ctx, rw, format, input, ty)),
        (None, Some(format)) => Some(encode(ctx, rw, format, input, ty)),
        (None, None) => None,
    }
}

/// `input`, codes of `format` held a byte each, as the floats of `ty`, through the `f32` each
/// converts to. Whole words of codes are read in place, the conversion naming the byte.
fn decode(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    format: Fp8Conversion,
    input: Value,
    ty: TypeHandle,
) -> Value {
    let lanes = lanes_of(ctx, input);
    let count = lanes.unwrap_or(1);
    let word_ty = i32_ty(ctx);
    let words = (lanes.is_some() && count.is_multiple_of(CODES_PER_WORD)).then(|| {
        let words = (count / CODES_PER_WORD) as u32;
        let words_ty = LlvmVectorType::get(ctx, word_ty, words, VectorTypeKind::Fixed).into();
        bitcast(ctx, rw, input, words_ty)
    });
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let convert = format!("llvm.amdgcn.cvt.f32.{}", format.name);
    let values: Vec<Value> = (0..count)
        .map(|at| {
            let (word, byte) = match words {
                Some(words) => (
                    extract_lane(ctx, rw, words, at / CODES_PER_WORD),
                    at % CODES_PER_WORD,
                ),
                None => {
                    let code = match lanes {
                        Some(_) => extract_lane(ctx, rw, input, at),
                        None => input,
                    };
                    let widen = llvm::ZExtOp::new_with_nneg(ctx, code, word_ty, false);
                    (insert(ctx, rw, &widen), 0)
                }
            };
            let byte = insert_i32_const(ctx, rw, byte as i32);
            call_intrinsic(ctx, rw, &convert, f32_ty, vec![word, byte])
        })
        .collect();
    let floats = match lanes {
        Some(_) => {
            let floats_ty =
                LlvmVectorType::get(ctx, f32_ty, count as u32, VectorTypeKind::Fixed).into();
            insert_vector_construct(ctx, rw, floats_ty, values)
        }
        None => values[0],
    };
    convert_float(ctx, rw, floats, ty)
}

/// `input`, `f32`s, as the codes of `format` in `ty`, rounded to nearest even and saturating at
/// its largest finite value. The conversion takes a value past the largest to the format's special
/// codes, so each is first brought within it; `maximum` and `minimum` pass a NaN on.
fn encode(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    format: Fp8Conversion,
    input: Value,
    ty: TypeHandle,
) -> Value {
    let smallest = bound(ctx, rw, input, -format.largest);
    let largest = bound(ctx, rw, input, format.largest);
    let floats_ty = input.get_type(ctx);
    let mangled = llvm_mangled_ty(ctx, floats_ty);
    let maximum = format!("llvm.maximum.{mangled}");
    let minimum = format!("llvm.minimum.{mangled}");
    let raised = call_intrinsic(ctx, rw, &maximum, floats_ty, vec![input, smallest]);
    let bounded = call_intrinsic(ctx, rw, &minimum, floats_ty, vec![raised, largest]);

    let convert = format!("llvm.amdgcn.cvt.pk.{}.f32", format.name);
    let lanes = lanes_of(ctx, input);
    let values: Vec<Value> = match lanes {
        Some(count) => (0..count)
            .map(|at| extract_lane(ctx, rw, bounded, at))
            .collect(),
        None => vec![bounded],
    };
    let words: Vec<Value> = values
        .chunks(CODES_PER_WORD)
        .map(|values| encode_word(ctx, rw, &convert, values))
        .collect();
    codes_of_words(ctx, rw, words, lanes, ty)
}

/// Up to four values as the codes of one word, the first in the low byte: each conversion fills
/// half of the word the one before it left, its second value repeating the first where missing.
fn encode_word(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    convert: &str,
    values: &[Value],
) -> Value {
    let word_ty = i32_ty(ctx);
    let empty = llvm::PoisonOp::new(ctx, word_ty);
    let empty = insert(ctx, rw, &empty);
    values
        .chunks(2)
        .enumerate()
        .fold(empty, |word, (half, pair)| {
            let upper = insert_bool_const(ctx, rw, half == 1);
            let (low, high) = (pair[0], pair[pair.len() - 1]);
            call_intrinsic(ctx, rw, convert, word_ty, vec![low, high, word, upper])
        })
}

/// The codes `words` hold, as the `lanes` of `ty`.
fn codes_of_words(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    words: Vec<Value>,
    lanes: Option<usize>,
    ty: TypeHandle,
) -> Value {
    let held = words.len() * CODES_PER_WORD;
    let word_ty = i32_ty(ctx);
    let words_ty =
        LlvmVectorType::get(ctx, word_ty, words.len() as u32, VectorTypeKind::Fixed).into();
    let byte_ty = int_ty(ctx, 8);
    let bytes_ty = LlvmVectorType::get(ctx, byte_ty, held as u32, VectorTypeKind::Fixed).into();
    let words = insert_vector_construct(ctx, rw, words_ty, words);
    let bytes = bitcast(ctx, rw, words, bytes_ty);
    match lanes {
        Some(count) if count == held => bytes,
        Some(count) => {
            let codes = (0..count)
                .map(|at| extract_lane(ctx, rw, bytes, at))
                .collect();
            insert_vector_construct(ctx, rw, ty, codes)
        }
        None => extract_lane(ctx, rw, bytes, 0),
    }
}

/// `value` in every lane of `like`, `f32`s.
fn bound(ctx: &mut Context, rw: &mut DialectConversionRewriter, like: Value, value: f32) -> Value {
    let attr = pliron::attribute::boxed_attr_cast(FPSingleAttr::from(value).into())
        .expect("an f32 constant");
    let constant = llvm::ConstantOp::new(ctx, attr);
    let scalar = insert(ctx, rw, &constant);
    match lanes_of(ctx, like) {
        Some(lanes) => insert_splat(ctx, rw, like.get_type(ctx), scalar, lanes),
        None => scalar,
    }
}

/// The lanes of a vector, `None` for a scalar.
fn lanes_of(ctx: &Context, value: Value) -> Option<usize> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements() as usize)
}
