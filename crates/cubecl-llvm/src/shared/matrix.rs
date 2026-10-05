//! Target-specific matrix lowering.

use crate::prelude::*;
use cubecl_core::{
    ir::{
        dialect::matrix::{
            CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp,
            RowIndexOp, StMatrixOp, StoreOp,
        },
        types::matrix::MatrixType,
    },
    prelude::{polyfills, *},
};

#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; a runtime that reaches it here has advertised \
     a matrix feature it cannot honour"
)]
pub struct MatrixOpUnsupported(LlvmTarget, &'static str);

#[type_interface_impl]
impl CubeToLLVMType for MatrixType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        match ctx.target() {
            #[cfg(feature = "amdgpu")]
            LlvmTarget::AmdGpu => crate::amdgpu::matrix::fragment_ty(ctx, self),
            #[cfg(feature = "nvptx")]
            LlvmTarget::Nvptx => crate::nvptx::matrix::fragment_ty(ctx, self),
            LlvmTarget::Cpu => {
                unimplemented!("the CPU target has no matrix fragments")
            }
        }
    }
}

macro_rules! dispatch_matrix_op {
    ($cube_op:ty, $method:ident) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                _rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                match ctx.target() {
                    #[cfg(feature = "amdgpu")]
                    LlvmTarget::AmdGpu => {
                        crate::amdgpu::matrix::$method(self, ctx, _rewriter, _operands_info)
                    }
                    #[cfg(feature = "nvptx")]
                    LlvmTarget::Nvptx => {
                        crate::nvptx::matrix::$method(self, ctx, _rewriter, _operands_info)
                    }
                    target => input_err!(
                        self.loc(ctx),
                        MatrixOpUnsupported(target, stringify!($cube_op))
                    ),
                }
            }
        }
    };
}

dispatch_matrix_op!(FillOp, fill);
dispatch_matrix_op!(LoadOp, load);
dispatch_matrix_op!(StoreOp, store);
dispatch_matrix_op!(MultiplyAccumulateOp, multiply_accumulate);
dispatch_matrix_op!(CastOp, cast);
dispatch_matrix_op!(RowIndexOp, row_index);
dispatch_matrix_op!(ColIndexOp, col_index);
dispatch_matrix_op!(MmaManualOp, mma_manual);
dispatch_matrix_op!(LdMatrixOp, ld_matrix);
dispatch_matrix_op!(StMatrixOp, st_matrix);

/// NVPTX matrix coordinates use the shared MMA polyfills.
macro_rules! lower_axis_index_polyfill {
    ($cube_op:ty, $formula:path) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn should_lower(&self, ctx: &Context) -> bool {
                match ctx.target() {
                    #[cfg(feature = "nvptx")]
                    LlvmTarget::Nvptx => true,
                    _ => false,
                }
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
                let elems_per_reg = 32 / matrix.unpacked_elem_size_bits(scope.ctx());
                let lane_id = self.lane_id(scope.ctx());
                let i = self.i(scope.ctx());

                let index = $formula(scope, lane_id.into(), i.into(), elems_per_reg, matrix.ident);
                vec![index.value(scope)]
            }
        }
    };
}

lower_axis_index_polyfill!(RowIndexOp, polyfills::mma::row_index::expand);
lower_axis_index_polyfill!(ColIndexOp, polyfills::mma::col_index::expand);

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) fn registers_as_vector(
    ctx: &Context,
    info: &OperandsInfo,
    value: Value,
) -> (TypeHandle, TypeHandle) {
    let array = info
        .lookup_operand_history(value)
        .into_iter()
        .rev()
        .chain(core::iter::once(value.get_type(ctx)))
        .find_map(|ty| {
            let ty = ty.deref(ctx);
            if let Some(array) = ty.downcast_ref::<CubeArrayType>() {
                return Some(*array);
            }
            let ptr = ty.downcast_ref::<CubePointerType>()?;
            let inner = ptr.inner.deref(ctx);
            inner.downcast_ref::<CubeArrayType>().copied()
        })
        .expect("a manual matrix operand is an array of registers");

    let (scalar, per_register) = match array.inner.deref(ctx).downcast_ref::<CubeVectorType>() {
        Some(vector) => (vector.inner, vector.vectorization),
        None => (array.inner, 1),
    };
    let elem = cube_type_to_llvm(ctx, scalar);
    let lanes = (array.length * per_register) as u32;
    let vector = LlvmVectorType::get(ctx, elem, lanes, VectorTypeKind::Fixed).into();
    (vector, scalar)
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) fn registers_value(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    vector_ty: TypeHandle,
) -> Value {
    let ty = value.get_type(ctx);
    if ty.deref(ctx).is::<LlvmPointerType>() {
        let op = llvm::LoadOp::new(ctx, value, vector_ty);
        return insert(ctx, rw, &op);
    }

    let (count, packed) = {
        let ty = ty.deref(ctx);
        let array = ty
            .downcast_ref::<LlvmArrayType>()
            .expect("registers are an array");
        (array.size(), array.elem_type())
    };
    let per_register = match packed.deref(ctx).downcast_ref::<LlvmVectorType>() {
        Some(vector) => vector.num_elements() as u64,
        None => 1,
    };

    let poison = llvm::PoisonOp::new(ctx, vector_ty);
    let mut acc = insert(ctx, rw, &poison);

    for register in 0..count {
        let op = llvm::ExtractValueOp::new(ctx, value, vec![register as u32])
            .expect("a constant index into the register array");
        let element = insert(ctx, rw, &op);

        for lane in 0..per_register {
            let value = if per_register == 1 {
                element
            } else {
                let from = insert_i32_const(ctx, rw, lane as i32);
                let op = llvm::ExtractElementOp::new(ctx, element, from);
                insert(ctx, rw, &op)
            };
            let to = insert_i32_const(ctx, rw, (register * per_register + lane) as i32);
            let op = llvm::InsertElementOp::new(ctx, acc, value, to);
            acc = insert(ctx, rw, &op);
        }
    }
    acc
}

#[cfg(feature = "nvptx")]
pub(crate) fn registers_array_ty(ctx: &Context, info: &OperandsInfo, value: Value) -> TypeHandle {
    let array = info
        .lookup_operand_history(value)
        .into_iter()
        .rev()
        .chain(core::iter::once(value.get_type(ctx)))
        .find_map(|ty| {
            let ty = ty.deref(ctx);
            if let Some(array) = ty.downcast_ref::<CubeArrayType>() {
                return Some(*array);
            }
            let ptr = ty.downcast_ref::<CubePointerType>()?;
            let inner = ptr.inner.deref(ctx);
            inner.downcast_ref::<CubeArrayType>().copied()
        })
        .expect("a manual matrix operand is an array of registers");
    let elem = cube_type_to_llvm(ctx, array.inner);
    LlvmArrayType::get(ctx, elem, array.length as u64).into()
}

/// Manual matrix outputs must preserve the frontend array type.
#[cfg(feature = "nvptx")]
pub(crate) fn vector_into_array(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    array_ty: TypeHandle,
) -> Value {
    let (count, elem_ty) = {
        let ty = array_ty.deref(ctx);
        let array = ty
            .downcast_ref::<LlvmArrayType>()
            .expect("registers are an array");
        (array.size() as usize, array.elem_type())
    };
    let per_element = match elem_ty.deref(ctx).downcast_ref::<LlvmVectorType>() {
        Some(vector) => vector.num_elements() as usize,
        None => 1,
    };

    let poison = llvm::PoisonOp::new(ctx, array_ty);
    let mut acc = insert(ctx, rw, &poison);

    for index in 0..count {
        let element = if per_element == 1 {
            let at = insert_i32_const(ctx, rw, index as i32);
            let op = llvm::ExtractElementOp::new(ctx, vector, at);
            insert(ctx, rw, &op)
        } else {
            let poison = llvm::PoisonOp::new(ctx, elem_ty);
            let mut packed = insert(ctx, rw, &poison);
            for lane in 0..per_element {
                let from = insert_i32_const(ctx, rw, (index * per_element + lane) as i32);
                let op = llvm::ExtractElementOp::new(ctx, vector, from);
                let value = insert(ctx, rw, &op);
                let to = insert_i32_const(ctx, rw, lane as i32);
                let op = llvm::InsertElementOp::new(ctx, packed, value, to);
                packed = insert(ctx, rw, &op);
            }
            packed
        };
        let op = llvm::InsertValueOp::new(ctx, acc, element, vec![index as u32]);
        acc = insert(ctx, rw, &op);
    }
    acc
}

/// Converts the lanes of a fragment between two float element types. `bf16` is carried as an
/// `i16`, which `fpext` and `fptrunc` cannot read, so it converts on its bit pattern through
/// `f32`. This is the arithmetic of the scalar `bf16_bits_to_f32` and `f32_to_bf16_bits` in
/// cubecl-core, written again in LLVM operations because a fragment stays opaque until this
/// lowering: the two must round alike.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) fn convert_lanes(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    from: TypeHandle,
    to: TypeHandle,
) -> Value {
    use cubecl_core::ir::types::scalar::Float32Type;

    if from == to {
        return value;
    }
    let lanes = {
        let ty = value.get_type(ctx);
        let ty = ty.deref(ctx);
        ty.downcast_ref::<LlvmVectorType>()
            .expect("a fragment is held in a vector")
            .num_elements()
    };
    let f32_ty: TypeHandle = Float32Type::get(ctx).into();
    let (value, from) = if from.is_bfloat16(ctx) {
        (bf16_lanes_to_f32(ctx, rw, value, lanes), f32_ty)
    } else {
        (value, from)
    };
    if to.is_bfloat16(ctx) {
        let wide = resize_float_lanes(ctx, rw, value, from, f32_ty, lanes);
        return f32_lanes_to_bf16(ctx, rw, wide, lanes);
    }
    resize_float_lanes(ctx, rw, value, from, to, lanes)
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn resize_float_lanes(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    from: TypeHandle,
    to: TypeHandle,
    lanes: u32,
) -> Value {
    let (from_bits, to_bits) = (from.size_bits(ctx), to.size_bits(ctx));
    let to_ty = lanes_ty(ctx, cube_type_to_llvm(ctx, to), lanes);
    if from_bits == to_bits {
        debug_assert_eq!(
            cube_type_to_llvm(ctx, from),
            cube_type_to_llvm(ctx, to),
            "a cast of the same width between two distinct LLVM types needs a conversion"
        );
        return value;
    }
    let op = if from_bits > to_bits {
        let op = llvm::FPTruncOp::new(ctx, value, to_ty);
        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        op.get_operation()
    } else {
        let op = llvm::FPExtOp::new(ctx, value, to_ty);
        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        op.get_operation()
    };
    rw.insert_operation(ctx, op);
    op.deref(ctx).get_result(0)
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn lanes_ty(ctx: &Context, elem: TypeHandle, lanes: u32) -> TypeHandle {
    LlvmVectorType::get(ctx, elem, lanes, VectorTypeKind::Fixed).into()
}

/// `lanes` 32-bit words, the integer shape of `lanes` `f32`.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn words_ty(ctx: &Context, lanes: u32) -> TypeHandle {
    lanes_ty(
        ctx,
        IntegerType::get(ctx, 32, Signedness::Signless).into(),
        lanes,
    )
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn word_splat(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: u32,
    lanes: u32,
) -> Value {
    let words_ty = words_ty(ctx, lanes);
    let scalar = insert_i32_const(ctx, rw, value as i32);
    insert_splat(ctx, rw, words_ty, scalar, lanes as usize)
}

/// `bf16` is the top half of an `f32`, so widening is exact.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn bf16_lanes_to_f32(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    halves: Value,
    lanes: u32,
) -> Value {
    use crate::shared::plane::{bitcast, shl};

    let words_ty = words_ty(ctx, lanes);
    let op = llvm::ZExtOp::new_with_nneg(ctx, halves, words_ty, false);
    let words = insert(ctx, rw, &op);
    let shift = word_splat(ctx, rw, 16, lanes);
    let bits = shl(ctx, rw, words, shift);
    let floats_ty = lanes_ty(ctx, FP32Type::get(ctx).into(), lanes);
    bitcast(ctx, rw, bits, floats_ty)
}

/// Round to nearest even; a NaN stays a NaN, quieted, with its sign.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn f32_lanes_to_bf16(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    floats: Value,
    lanes: u32,
) -> Value {
    use crate::shared::plane::{add, and, bitcast, icmp, lshr, or, select};

    let words_ty = words_ty(ctx, lanes);
    let bits = bitcast(ctx, rw, floats, words_ty);
    let shift = word_splat(ctx, rw, 16, lanes);
    let one = word_splat(ctx, rw, 1, lanes);
    let below_half = word_splat(ctx, rw, 0x7FFF, lanes);
    let magnitude_mask = word_splat(ctx, rw, u32::MAX >> 1, lanes);
    let infinity = word_splat(ctx, rw, f32::INFINITY.to_bits(), lanes);
    let quiet = word_splat(ctx, rw, 1 << 6, lanes);

    let truncated = lshr(ctx, rw, bits, shift);
    let lsb = and(ctx, rw, truncated, one);
    let biased = add(ctx, rw, bits, below_half);
    let biased = add(ctx, rw, biased, lsb);
    let rounded = lshr(ctx, rw, biased, shift);

    let magnitude = and(ctx, rw, bits, magnitude_mask);
    let is_nan = icmp(ctx, rw, ICmpPredicateAttr::UGT, magnitude, infinity);
    let quieted = or(ctx, rw, truncated, quiet);
    let code = select(ctx, rw, is_nan, quieted, rounded);

    let halves_ty = lanes_ty(
        ctx,
        IntegerType::get(ctx, 16, Signedness::Signless).into(),
        lanes,
    );
    let op = llvm::TruncOp::new(ctx, code, halves_ty);
    insert(ctx, rw, &op)
}
