//! The matrix operations, dispatched to the target that has the instructions for them.
//!
//! Nothing about a matrix fragment is shared between the two GPUs: AMD spreads a documented
//! layout across the wavefront and `CubeCL`'s lowering addresses it element by element, while
//! NVIDIA's WMMA fragment is opaque and only its own load and store instructions know where
//! anything sits. So unlike [`plane`](super::plane), where the targets agree on primitives and
//! differ only in the instructions, there is no shared body here worth writing -- each op is
//! forwarded whole.
//!
//! What is shared is that these ops exist in one place, so a target that does not implement
//! one says so rather than falling through to another target's instructions.

use cubecl_core::ir::Scope;
use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp,
    StMatrixOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use cubecl_core::prelude::polyfills;
use cubecl_core::prelude::*;
use pliron::input_err;
use thiserror::Error;

use crate::shared::polyfill::LowerOp;
use crate::shared::to_llvm::prelude::*;
use crate::target::{CtxTarget, LlvmTarget};
use cubecl_core::ir::types::ArrayType as CubeArrayType;

/// A matrix operation on a target with no matrix instructions.
#[derive(Debug, Error)]
#[error(
    "the {0:?} target has no lowering for `{1}`; a runtime that reaches it here has advertised \
     a matrix feature it cannot honour"
)]
pub struct MatrixOpUnsupported(LlvmTarget, &'static str);

/// The LLVM value a fragment of `matrix` lives in.
///
/// Both targets hold one in a vector, which is what lets the fragment be an `alloca` the other
/// ops load and store; what differs is how many elements of it a lane holds.
#[type_interface_impl]
impl CubeToLLVMType for MatrixType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        match ctx.target() {
            LlvmTarget::AmdGpu => crate::amdgpu::matrix::fragment_ty(ctx, self),
            LlvmTarget::Nvptx => crate::nvptx::matrix::fragment_ty(ctx, self),
            // Reached only if a CPU kernel declares a matrix, which needs a device advertising
            // a matrix feature; the CPU advertises none. A type conversion cannot report an
            // error, so this is the one place the refusal has to be a panic.
            LlvmTarget::Cpu => {
                unimplemented!("the CPU target has no matrix fragments")
            }
        }
    }
}

/// Forwards one matrix op to the target that has the instructions for it.
macro_rules! dispatch_matrix_op {
    ($cube_op:ty, $method:ident) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                operands_info: &OperandsInfo,
            ) -> Result<()> {
                match ctx.target() {
                    LlvmTarget::AmdGpu => {
                        crate::amdgpu::matrix::$method(self, ctx, rewriter, operands_info)
                    }
                    LlvmTarget::Nvptx => {
                        crate::nvptx::matrix::$method(self, ctx, rewriter, operands_info)
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

/// Answers `row_index` and `col_index` from the documented `mma.sync` layout.
///
/// A polyfill rather than a dialect conversion, because the answer is arithmetic on the lane
/// index that `CubeCL` already writes once, in
/// [`polyfills::mma`](cubecl_core::prelude::polyfills::mma) -- the same source the CUDA C++
/// backend expands. Every backend emitting `mma.sync` has to agree with the instruction about
/// which element of the tile a register holds, and one copy of the formulas is how that is
/// kept true.
///
/// Only NVPTX takes this route: AMD's fragment layout is its own, and it answers these in its
/// dialect conversion. Running before that conversion is what lets this be written in `CubeCL`
/// at all, since the ops it expands to have not been lowered yet.
macro_rules! lower_axis_index_polyfill {
    ($cube_op:ty, $formula:path) => {
        #[op_interface_impl]
        impl LowerOp for $cube_op {
            fn should_lower(&self, ctx: &Context) -> bool {
                ctx.target() == LlvmTarget::Nvptx
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
                // How many elements share a 32 bit register, which is what turns an element
                // index into a register index in the formulas.
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

/// The LLVM vector holding the registers `value` points at.
///
/// The manual ops carry their operands as arrays rather than fragments, but the registers are
/// the same ones, so the array is read as the vector the instruction expects. Neither target
/// has a say in this -- the array is the frontend's shape, not the hardware's -- so both read
/// it the same way.
pub(crate) fn registers_as_vector(
    ctx: &Context,
    info: &OperandsInfo,
    value: Value,
) -> (TypeHandle, TypeHandle) {
    // The inputs are array values and the output a pointer to one, so both shapes are looked
    // for and the registers read accordingly.
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

    // The registers are packed as vectors, so the array is flattened into the one vector the
    // instruction takes.
    let (scalar, per_register) = match array.inner.deref(ctx).downcast_ref::<CubeVectorType>() {
        Some(vector) => (vector.inner, vector.vectorization),
        None => (array.inner, 1),
    };
    let elem = cube_type_to_llvm(ctx, scalar);
    let lanes = (array.length * per_register) as u32;
    let vector = LlvmVectorType::get(ctx, elem, lanes, VectorTypeKind::Fixed).into();
    (vector, scalar)
}

/// The registers of `value` as the one vector the instruction takes.
///
/// The registers arrive as an array, of vectors where several share a register. An array is not
/// a vector as far as a bitcast is concerned, so it is taken apart and rebuilt.
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

/// The LLVM array a manual operand's registers are held in, as the frontend shaped it.
///
/// The counterpart of [`registers_as_vector`]: that answers what the instruction wants, this
/// what the kernel has. Written back through [`vector_into_array`] so a destination keeps the
/// type its later uses were built against.
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

/// `vector` written back into the array shape `array_ty`.
///
/// Storing the flat vector into the destination instead would be bit-identical -- pointers are
/// opaque -- but it tells the conversion that the array type maps to a vector, and every other
/// value of that array type then gets rewritten to one, invalidating the `extractvalue`s built
/// against them. Writing the array keeps the two shapes apart.
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
