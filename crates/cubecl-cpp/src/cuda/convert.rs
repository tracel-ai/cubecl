//! Cuda conversion functions
#![allow(unused)]

use core::{fmt, ops::Deref};

use cubecl_core::{
    self as cubecl,
    ir::{
        dialect::general::CastOp,
        interfaces::{ScalarType, TypedExt},
        match_ty,
        prelude::*,
        types::{VectorType, scalar::*},
    },
    prelude::*,
};
use pliron::{printable::Printable, utils::apfloat::Float8E5M2};

use crate::{
    cuda::{cuda_op_with_out, ty::*},
    shared::{
        CppValue,
        lowering::LowerOp,
        ty::{TypeExt, TypeExtCPP, TypedExtCPP},
    },
    target::Cuda,
};

/// special cast function for recursive conversion in the case of minifloat to minifloat conversion
///
/// Needs to jump through a lot of hoops to deal with CUDA nonsense.
/// The overview of available conversions is as follows:
///
/// | From                     | To             | Extra args                 |
/// | ------------------------ | -------------- | -------------------------- |
/// | f16/bf16/f32/f64         | e4m3/e5m2      | Interpretation, saturation |
/// | f16/bf16/f32/f64         | e3m2/e2m3/e2m1 | Interpretation, rounding   |
/// | bf16/f32/f64             | e8m0           | saturation, rounding       |
/// | e4m3/e5m2/e3m2/e2m3/e2m1 | f16            | Interpretation,            |
/// | e8m0                     | bf16           |                            |
///
/// When the input and output don't match these options, we need to do a two-step conversion.
/// When the input is a minifloat we always need to cast out to `f16`/`bf16`, and then convert to
/// the actual out type if it differs. Trying to cast ints also requires an extra conversion, and
/// so does `f16` to `e8m0` (though it's not recommended to do that anyways, you should be using
/// `e5m2` for that since you don't have 8 bits of exponent in f16).
///
/// See also:
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html>
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html>
/// <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html>
#[op_interface_impl]
impl LowerOp<Cuda> for CastOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        let input = self.input(ctx);
        let out = self.get_result(ctx);
        let should_lower_from = (input.is_fp8_fp6_fp4(ctx) || input.is_float4x2(ctx))
            && intermediate_for_ty(ctx, input.get_type(ctx)) != out.get_type(ctx);
        let should_lower_to = (out.is_fp8_fp6_fp4(ctx) || out.is_float4x2(ctx))
            && intermediate_for_ty(ctx, out.get_type(ctx)) != input.get_type(ctx);
        should_lower_from || should_lower_to
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let mut current = self.input(ctx);
        let out_ty = self.get_result(ctx).get_type(ctx);
        if current.is_fp8_fp6_fp4(ctx) || current.is_float4x2(ctx) {
            let intermediate = intermediate_for_ty(ctx, current.get_type(ctx));
            current = cast_value(scope, current, intermediate);
        }
        if out_ty.is_fp8_fp6_fp4(ctx) || out_ty.is_float4x2(ctx) {
            current = cast_value(scope, current, intermediate_for_ty(ctx, out_ty));
        }
        vec![cast_value(scope, current, out_ty)]
    }
}

fn intermediate_for_ty(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    let vector_size = ty.vector_size(ctx);
    let intermediate = if ty.scalar_ty(ctx).deref(ctx).is::<Float8E8M0Type>() {
        BFloat16Type::get(ctx).to_handle()
    } else if ty.is_float4x2(ctx) {
        return VectorType::get(ctx, Float16Type::get(ctx).to_handle(), vector_size * 2)
            .to_handle();
    } else {
        Float16Type::get(ctx).to_handle()
    };
    if vector_size > 1 {
        VectorType::get(ctx, intermediate, vector_size).to_handle()
    } else {
        intermediate
    }
}

cuda_op_with_out!(CastOp, |op, ctx| {
    let input = op.input(ctx);
    let out_ty = op.get_result(ctx).get_type(ctx);
    if input.is_fp8_fp6_fp4(ctx) || input.is_packed_fp6_fp8_fp4(ctx) {
        cast_minifloat_to_half(ctx, input)
    } else if out_ty.is_fp8_fp6_fp4(ctx) || out_ty.is_packed_fp6_fp8_fp4(ctx) {
        cast_half_to_minifloat(ctx, input, out_ty)
    } else if out_ty.is_tfloat32(ctx) {
        format!("nvcuda::wmma::__float_to_tf32({})", input.name(ctx))
    } else {
        format!("{}({})", out_ty.to_cpp(ctx), input.name(ctx))
    }
});

// Cast from minifloat to half/bf16. Could be made more generic, but a simple mapping is easier
// to understand. The naming is very inconsistent (i.e. halfraw2 vs bf162raw)
fn cast_minifloat_to_half(ctx: &Context, input: Value) -> String {
    let in_ty = input.get_type(ctx).deref(ctx);
    let in_val = input.name(ctx);
    match_ty!((in_ty) {
        Float8E8M0Type => format!("__nv_bfloat16(__nv_cvt_e8m0_to_bf16raw({in_val}))"),
        Float8E8M0x2Type => format!("__nv_bfloat162(__nv_cvt_e8m0x2_to_bf162raw({in_val}))"),
        Float8E4M3Type => format!("__half(__nv_cvt_fp8_to_halfraw({in_val}, __NV_E4M3))"),
        Float8E4M3x2Type => format!("__half2(__nv_cvt_fp8x2_to_halfraw2({in_val}, __NV_E4M3))"),
        Float8E5M2Type => format!("__half(__nv_cvt_fp8_to_halfraw({in_val}, __NV_E5M2))"),
        Float8E5M2x2Type => format!("__half2(__nv_cvt_fp8x2_to_halfraw2({in_val}, __NV_E5M2))"),
        Float6E2M3Type => format!("__half(__nv_cvt_fp6_to_halfraw({in_val}, __NV_E2M3))"),
        Float6E2M3x2Type => format!("__half2(__nv_cvt_fp6x2_to_halfraw2({in_val}, __NV_E2M3))"),
        Float6E3M2Type => format!("__half(__nv_cvt_fp6_to_halfraw({in_val}, __NV_E3M2))"),
        Float6E3M2x2Type => format!("__half(__nv_cvt_fp6x2_to_halfraw2({in_val}, __NV_E3M2))"),
        Float4E2M1Type => format!("__half(__nv_cvt_fp4_to_halfraw({in_val}, __NV_E2M1))"),
        Float4E2M1x2Type => format!("__half2(__nv_cvt_fp4x2_to_halfraw2({in_val}, __NV_E2M1))"),;
        _ => panic!("Unsupported type {}", in_ty.display(ctx))
    })
}

// Cast to minifloat from half/bf16. Could be made more generic, but a simple mapping is easier
// to understand. The naming is very inconsistent (i.e. halfraw2 vs bf162raw)
fn cast_half_to_minifloat(ctx: &Context, input: Value, out_ty: TypeHandle) -> String {
    let in_val = input.name(ctx);
    match_ty!((out_ty.deref(ctx)) {
        Float8E8M0Type => format!("__nv_cvt_bfloat16raw_to_e8m0({in_val}, __NV_NOSAT, cudaRoundPosInf)"),
        Float8E8M0x2Type => format!("__nv_cvt_bfloat162raw_to_e8m0x2({in_val}, __NV_NOSAT, cudaRoundPosInf)"),
        Float8E4M3Type => format!("__nv_cvt_halfraw_to_fp8({in_val}, __NV_NOSAT, __NV_E4M3)"),
        Float8E4M3x2Type => format!("__nv_cvt_halfraw2_to_fp8x2({in_val}, __NV_NOSAT, __NV_E4M3)"),
        Float8E5M2Type => format!("__nv_cvt_halfraw_to_fp8({in_val}, __NV_NOSAT, __NV_E5M2)"),
        Float8E5M2x2Type => format!("__nv_cvt_halfraw2_to_fp8x2({in_val}, __NV_NOSAT, __NV_E5M2)"),
        Float6E2M3Type => format!("__nv_cvt_halfraw_to_fp6({in_val}, __NV_E2M3, cudaRoundNearest)"),
        Float6E2M3x2Type => format!("__nv_cvt_halfraw2_to_fp6x2({in_val}, __NV_E2M3, cudaRoundNearest)"),
        Float6E3M2Type => format!("__nv_cvt_halfraw_to_fp6({in_val}, __NV_E3M2, cudaRoundNearest)"),
        Float6E3M2x2Type => format!("__nv_cvt_halfraw2_to_fp6x2({in_val}, __NV_E3M2, cudaRoundNearest)"),
        Float4E2M1Type => format!("__nv_cvt_halfraw_to_fp4({in_val}, __NV_E2M1, cudaRoundNearest)"),
        Float4E2M1x2Type => format!("__nv_cvt_halfraw2_to_fp4x2({in_val}, __NV_E2M1, cudaRoundNearest)"),;
        _ => panic!("Unsupported type {}", out_ty.deref(ctx).display(ctx))
    })
}
