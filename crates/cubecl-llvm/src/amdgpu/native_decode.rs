//! AMDGPU's own decode of `ue8m0`, a bare exponent: the scale `2^(code - 127)` is one
//! `v_ldexp_f32` of `2^-127`, where the software codec spends two compares and two selects on the
//! edge codes, each compare a scalar register the selects then wait on. A kernel reading
//! block-scaled values decodes a scale per block, often beside matrix instructions that share
//! the same vector unit, so the decode's cost shows up in the product's time.

use crate::{
    prelude::*,
    shared::{
        intrinsic::call_intrinsic, native_decode::NativeDecode, to_llvm::general::convert_float,
    },
};

/// The value of `ue8m0`'s code `0`, `2^-127`: a subnormal `f32`, which the default denormal mode
/// of an AMDGPU kernel keeps, as the software codec's decode of that code does.
const UE8M0_SMALLEST: f32 = f32::from_bits(1 << 22);

/// `input`, `ue8m0` codes held a byte each, as the floats of `ty`: `ldexp(2^-127, code)`, the
/// bias folded into the base, exact for every code but the last; the last, `255`, is the format's
/// NaN, which `ldexp` lands on infinity and `fma(0, s, s)` turns into NaN (`0 · ∞`) while leaving
/// every finite scale as it is.
pub(crate) fn decode(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    format: NativeDecode,
    input: Value,
    ty: TypeHandle,
) -> Value {
    assert_eq!(
        format,
        NativeDecode::UE8M0,
        "an fp8 decode is the part's own conversion (`fp8::cast`), lowered before this"
    );
    let lanes = input
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements());
    let lanes_of = |ctx: &mut Context, scalar: TypeHandle| -> TypeHandle {
        match lanes {
            Some(lanes) => LlvmVectorType::get(ctx, scalar, lanes, VectorTypeKind::Fixed).into(),
            None => scalar,
        }
    };
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let f32_scalar: TypeHandle = FP32Type::get(ctx).into();
    let (ints_ty, floats_ty) = (lanes_of(ctx, i32_ty), lanes_of(ctx, f32_scalar));
    let splat =
        |ctx: &mut Context, rw: &mut DialectConversionRewriter, value: Value, ty| match lanes {
            Some(lanes) => insert_splat(ctx, rw, ty, value, lanes as usize),
            None => value,
        };

    let widen = llvm::ZExtOp::new_with_nneg(ctx, input, ints_ty, false);
    let code = insert(ctx, rw, &widen);

    let smallest = f32_const(ctx, rw, UE8M0_SMALLEST);
    let smallest = splat(ctx, rw, smallest, floats_ty);
    let ldexp = format!(
        "llvm.ldexp.{}.{}",
        llvm_mangled_ty(ctx, floats_ty),
        llvm_mangled_ty(ctx, ints_ty)
    );
    let scale = call_intrinsic(ctx, rw, &ldexp, floats_ty, vec![smallest, code]);

    let zero = f32_const(ctx, rw, 0.0);
    let zero = splat(ctx, rw, zero, floats_ty);
    let fma = format!("llvm.fma.{}", llvm_mangled_ty(ctx, floats_ty));
    let scale = call_intrinsic(ctx, rw, &fma, floats_ty, vec![zero, scale, scale]);
    convert_float(ctx, rw, scale, ty)
}

fn f32_const(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: f32) -> Value {
    let attr = pliron::attribute::boxed_attr_cast(
        pliron::builtin::attributes::FPSingleAttr::from(value).into(),
    )
    .expect("an f32 constant");
    let op = llvm::ConstantOp::new(ctx, attr);
    insert(ctx, rw, &op)
}
