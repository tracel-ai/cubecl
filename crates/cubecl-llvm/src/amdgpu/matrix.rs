//! Lowering of the matrix operations to the wavefront's WMMA instructions.
//!
//! A fragment is a vector held across the lanes of the wavefront: A and B hold the `k` range of
//! one row or column, the accumulator eight rows' worth. The layouts differ by generation —
//! RDNA3 gives both halves of the wave the whole `k` range and pads 16 bit accumulators out to
//! 32 bits per element, RDNA4 splits `k` between the halves and packs densely — so the element
//! counts and the index arithmetic are derived from [`WmmaGeneration`] rather than fixed.

use cubecl_core::ir::ContextExt;
use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use cubecl_core::ir::types::{MatrixIdent, MatrixLayout, MatrixShape};

use crate::amdgpu::plane::lane_id;
use crate::shared::to_llvm::prelude::*;

/// Which WMMA the device has. The instructions are the same shape; the fragments are not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmmaGeneration {
    /// gfx11.
    Rdna3,
    /// gfx12.
    Rdna4,
}

impl WmmaGeneration {
    /// Elements of A or B each lane holds. RDNA3 gives every lane the whole `k` range and
    /// duplicates it across the halves of the wave; RDNA4 gives each half its own.
    pub fn ab_elems(&self, k: usize) -> usize {
        match self {
            WmmaGeneration::Rdna3 => k,
            WmmaGeneration::Rdna4 => k / 2,
        }
    }

    /// Whether a 16 bit accumulator is padded to one element per 32 bit register.
    pub fn pads_half_accumulator(&self) -> bool {
        *self == WmmaGeneration::Rdna3
    }

    /// The generation of `arch`, or `None` where there is no WMMA at all.
    pub fn of(arch: &str) -> Option<Self> {
        if arch.starts_with("gfx11") {
            Some(WmmaGeneration::Rdna3)
        } else if arch.starts_with("gfx12") {
            Some(WmmaGeneration::Rdna4)
        } else {
            None
        }
    }
}

impl CtxWmma for Context {}

/// The WMMA generation on the context.
pub trait CtxWmma: ContextExt {
    fn wmma(&self) -> WmmaGeneration {
        *self
            .aux_ty::<Option<WmmaGeneration>>()
            .as_ref()
            .expect("matrix ops are only compiled for devices that have WMMA")
    }
    fn set_wmma(&mut self, generation: Option<WmmaGeneration>) {
        self.set_aux_ty(generation);
    }
}

/// Whether `elem` is one of the 16 bit floats a fragment can hold.
fn is_half(ctx: &Context, elem: TypeHandle) -> bool {
    elem.is_float16(ctx) || elem.is_bfloat16(ctx)
}

/// Eight registers hold the accumulator, whatever it holds.
const ACCUMULATOR_REGISTERS: usize = 8;

/// Lanes over which one row or column of a fragment is spread.
const LANES_PER_ROW: u32 = 16;

/// How many elements of `matrix` one lane holds, and how far apart they sit in the fragment.
///
/// A 16 bit accumulator on RDNA3 occupies the low half of each register and leaves the other
/// half alone, so its elements are every second one.
fn fragment_layout(ctx: &Context, matrix: &MatrixType) -> (usize, usize) {
    let generation = ctx.wmma();
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => (generation.ab_elems(matrix.shape.k), 1),
        MatrixIdent::Accumulator => {
            let padded = is_half(ctx, matrix.elem_ty) && generation.pads_half_accumulator();
            (ACCUMULATOR_REGISTERS, if padded { 2 } else { 1 })
        }
    }
}

/// The LLVM vector a fragment of `matrix` lives in.
fn fragment_ty(ctx: &Context, matrix: &MatrixType) -> TypeHandle {
    let (elems, step) = fragment_layout(ctx, matrix);
    let elem = cube_type_to_llvm(ctx, matrix.elem_ty);
    LlvmVectorType::get(ctx, elem, (elems * step) as u32, VectorTypeKind::Fixed).into()
}

#[type_interface_impl]
impl CubeToLLVMType for MatrixType {
    fn convert(&self, ctx: &Context) -> TypeHandle {
        fragment_ty(ctx, self)
    }
}

/// The `MatrixType` a matrix operand points at.
///
/// The conversion has already turned the pointer opaque by the time these ops are rewritten, so
/// the fragment's shape is read out of the operand's type history rather than its current type.
fn matrix_of(ctx: &Context, info: &OperandsInfo, value: Value) -> MatrixType {
    let pointee = info
        .lookup_operand_history(value)
        .into_iter()
        .rev()
        .chain(core::iter::once(value.get_type(ctx)))
        .find_map(|ty| {
            let ty = ty.deref(ctx);
            let ptr = ty.downcast_ref::<CubePointerType>()?;
            let pointee = ptr.inner.deref(ctx);
            pointee.downcast_ref::<MatrixType>().copied()
        });
    pointee.expect("a matrix operand points at a matrix")
}

/// `lhs * rhs` over `i32`.
fn mul(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    rw.insert_op(ctx, &op);
    op.get_result(ctx)
}

/// `lhs + rhs` over `i32`.
fn add(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    rw.insert_op(ctx, &op);
    op.get_result(ctx)
}

/// Where in the fragment's row or column a lane sits, and which half of the wave it is in.
#[derive(Clone, Copy)]
struct LanePosition {
    in_row: Value,
    half: Value,
}

/// This lane's position within its wavefront.
fn lane_position(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> LanePosition {
    let lane = lane_id(ctx, rw);
    let width = insert_i32_const(ctx, rw, LANES_PER_ROW as i32);

    let in_row = llvm::URemOp::new(ctx, lane, width);
    rw.insert_op(ctx, &in_row);
    let half = llvm::UDivOp::new(ctx, lane, width);
    rw.insert_op(ctx, &half);

    LanePosition {
        in_row: in_row.get_result(ctx),
        half: half.get_result(ctx),
    }
}

/// The index into the backing memory of this lane's `i`th element of `matrix`.
///
/// A and B walk the `k` range of one row or column; RDNA4 splits that range between the halves
/// of the wave, so the half selects which part this lane holds. The accumulator instead walks
/// eight rows two apart, the half choosing between them.
fn element_index(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    layout: MatrixLayout,
    i: usize,
    lane: LanePosition,
    stride: Value,
) -> Value {
    let LanePosition { in_row, half } = lane;
    let generation = ctx.wmma();

    let (along, across) = match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => {
            let step = insert_i32_const(ctx, rw, i as i32);
            let along = match generation {
                WmmaGeneration::Rdna3 => step,
                WmmaGeneration::Rdna4 => {
                    let per_half = insert_i32_const(ctx, rw, matrix.shape.k as i32 / 2);
                    let offset = mul(ctx, rw, half, per_half);
                    add(ctx, rw, step, offset)
                }
            };
            (along, in_row)
        }
        MatrixIdent::Accumulator => {
            // RDNA3 interleaves the halves row by row, RDNA4 gives each half a contiguous
            // block of eight.
            let along = match generation {
                WmmaGeneration::Rdna3 => {
                    let row = insert_i32_const(ctx, rw, (i * 2) as i32);
                    add(ctx, rw, row, half)
                }
                WmmaGeneration::Rdna4 => {
                    let row = insert_i32_const(ctx, rw, i as i32);
                    let block = insert_i32_const(ctx, rw, ACCUMULATOR_REGISTERS as i32);
                    let offset = mul(ctx, rw, half, block);
                    add(ctx, rw, row, offset)
                }
            };
            (along, in_row)
        }
    };

    // Whichever of the two the layout makes contiguous gets the stride.
    let strided = matches!(
        (matrix.ident, layout),
        (MatrixIdent::A, MatrixLayout::ColMajor)
            | (MatrixIdent::B, MatrixLayout::RowMajor)
            | (MatrixIdent::Accumulator, MatrixLayout::RowMajor)
    );
    if strided {
        let scaled = mul(ctx, rw, along, stride);
        add(ctx, rw, scaled, across)
    } else {
        let scaled = mul(ctx, rw, across, stride);
        add(ctx, rw, along, scaled)
    }
}

/// The address of element `index` of `base`, whose elements are `elem_ty`.
fn element_ptr(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    base: Value,
    index: Value,
    elem_ty: TypeHandle,
) -> Value {
    let gep = llvm::GetElementPtrOp::new(ctx, base, vec![llvm::GepIndex::Value(index)], elem_ty);
    rw.insert_op(ctx, &gep);
    gep.get_result(ctx)
}

/// Loads the fragment `matrix` points at.
fn load_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::LoadOp::new(ctx, matrix, ty);
    rw.insert_op(ctx, &op);
    op.get_result(ctx)
}

/// Stores `value` into the fragment `matrix` points at.
fn store_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    value: Value,
) {
    let op = llvm::StoreOp::new(ctx, value, matrix);
    rw.insert_op(ctx, &op);
}

#[op_interface_impl]
impl ToLLVMDialect for FillOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let matrix = self.matrix(ctx);
        let value = self.value(ctx);

        let ty = matrix_of(ctx, operands_info, matrix);
        let (elems, step) = fragment_layout(ctx, &ty);
        let frag_ty = fragment_ty(ctx, &ty);
        let lanes = elems * step;

        let filled = insert_splat(ctx, rw, frag_ty, value, lanes);
        store_fragment(ctx, rw, matrix, filled);

        rw.erase_operation(ctx, old_op);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for LoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let matrix = self.matrix(ctx);
        let source = self.source(ctx);
        let stride = self.stride(ctx);
        let layout = self.layout(ctx).0;

        let ty = matrix_of(ctx, operands_info, matrix);
        let (elems, step) = fragment_layout(ctx, &ty);
        let frag_ty = fragment_ty(ctx, &ty);
        let elem_ty = cube_type_to_llvm(ctx, ty.elem_ty);
        let lane = lane_position(ctx, rw);

        let poison = llvm::PoisonOp::new(ctx, frag_ty);
        rw.insert_op(ctx, &poison);
        let mut frag = poison.get_result(ctx);

        for i in 0..elems {
            let index = element_index(ctx, rw, &ty, layout, i, lane, stride);
            let ptr = element_ptr(ctx, rw, source, index, elem_ty);
            let value = load_fragment(ctx, rw, ptr, elem_ty);

            let slot = insert_i32_const(ctx, rw, (i * step) as i32);
            let insert = llvm::InsertElementOp::new(ctx, frag, value, slot);
            rw.insert_op(ctx, &insert);
            frag = insert.get_result(ctx);
        }

        store_fragment(ctx, rw, matrix, frag);
        rw.erase_operation(ctx, old_op);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for StoreOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let matrix = self.matrix(ctx);
        let destination = self.destination(ctx);
        let stride = self.stride(ctx);
        let layout = self.layout(ctx).0;

        let ty = matrix_of(ctx, operands_info, matrix);
        let (elems, step) = fragment_layout(ctx, &ty);
        let frag_ty = fragment_ty(ctx, &ty);
        let elem_ty = cube_type_to_llvm(ctx, ty.elem_ty);
        let lane = lane_position(ctx, rw);

        let frag = load_fragment(ctx, rw, matrix, frag_ty);

        for i in 0..elems {
            let slot = insert_i32_const(ctx, rw, (i * step) as i32);
            let extract = llvm::ExtractElementOp::new(ctx, frag, slot);
            rw.insert_op(ctx, &extract);

            let index = element_index(ctx, rw, &ty, layout, i, lane, stride);
            let ptr = element_ptr(ctx, rw, destination, index, elem_ty);
            store_fragment(ctx, rw, ptr, extract.get_result(ctx));
        }

        rw.erase_operation(ctx, old_op);
        Ok(())
    }
}

/// The WMMA format name of a fragment element type.
fn wmma_format(ctx: &Context, elem: TypeHandle) -> &'static str {
    if elem.is_float32(ctx) {
        "f32"
    } else if elem.is_bfloat16(ctx) {
        "bf16"
    } else if elem.is_float16(ctx) {
        "f16"
    } else {
        panic!("no WMMA takes this element type")
    }
}

#[op_interface_impl]
impl ToLLVMDialect for MultiplyAccumulateOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let (a, b, c, d) = (
            self.mat_a(ctx),
            self.mat_b(ctx),
            self.mat_c(ctx),
            self.mat_d(ctx),
        );

        let a_ty = matrix_of(ctx, operands_info, a);
        let c_ty = matrix_of(ctx, operands_info, c);
        let ab_frag_ty = fragment_ty(ctx, &a_ty);
        let cd_frag_ty = fragment_ty(ctx, &c_ty);

        let a_val = load_fragment(ctx, rw, a, ab_frag_ty);
        let b_val = load_fragment(ctx, rw, b, ab_frag_ty);
        let c_val = load_fragment(ctx, rw, c, cd_frag_ty);

        let MatrixShape { m, n, k } = a_ty.shape;
        let ab = wmma_format(ctx, a_ty.elem_ty);
        let cd = wmma_format(ctx, c_ty.elem_ty);
        // The intrinsic is overloaded on both fragment types, so the name carries them.
        let name = format!(
            "llvm.amdgcn.wmma.{cd}.{m}x{n}x{k}.{ab}.{}.{}",
            llvm_mangled_ty(ctx, cd_frag_ty),
            llvm_mangled_ty(ctx, ab_frag_ty),
        );

        let mut args = vec![a_val, b_val, c_val];
        let mut arg_tys = vec![ab_frag_ty, ab_frag_ty, cd_frag_ty];
        // RDNA3 writes a 16 bit result into one half of each register and takes an `opsel`
        // saying which. RDNA4 packs them densely and has no such argument.
        if ctx.wmma().pads_half_accumulator() && is_half(ctx, c_ty.elem_ty) {
            let low_half = insert_bool_const(ctx, rw, false);
            arg_tys.push(low_half.get_type(ctx));
            args.push(low_half);
        }

        let fn_ty = FuncType::get(ctx, cd_frag_ty, arg_tys, false);
        let op = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
        rw.insert_op(ctx, &op);
        store_fragment(ctx, rw, d, op.get_result(ctx));

        rw.erase_operation(ctx, old_op);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for CastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let input = self.input(ctx);
        let output = self.output(ctx);

        let in_ty = matrix_of(ctx, operands_info, input);
        let out_ty = matrix_of(ctx, operands_info, output);
        let in_frag_ty = fragment_ty(ctx, &in_ty);
        let out_frag_ty = fragment_ty(ctx, &out_ty);

        let value = load_fragment(ctx, rw, input, in_frag_ty);

        // Both fragments hold the same elements, so this is one widening or narrowing over the
        // whole vector rather than anything lane-wise.
        let narrower = in_ty.elem_ty.size_bits(ctx) > out_ty.elem_ty.size_bits(ctx);
        let cast = if narrower {
            let op = llvm::FPTruncOp::new(ctx, value, out_frag_ty);
            op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
            rw.insert_op(ctx, &op);
            op.get_result(ctx)
        } else {
            let op = llvm::FPExtOp::new(ctx, value, out_frag_ty);
            op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
            rw.insert_op(ctx, &op);
            op.get_result(ctx)
        };
        store_fragment(ctx, rw, output, cast);

        rw.erase_operation(ctx, old_op);
        Ok(())
    }
}

/// Which of the two matrix axes an element sits on.
enum Axis {
    Row,
    Col,
}

/// The row or column of the logical matrix that this lane's `i`th element holds.
///
/// A holds a row of the `k` range, B a column of it, and the accumulator a row of the output, so
/// which axis the lane index names and which the element index names depends on the fragment.
fn axis_index(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    axis: Axis,
    lane: Value,
    i: Value,
) -> Value {
    let generation = ctx.wmma();
    let width = insert_i32_const(ctx, rw, LANES_PER_ROW as i32);

    let in_row = llvm::URemOp::new(ctx, lane, width);
    rw.insert_op(ctx, &in_row);
    let half_op = llvm::UDivOp::new(ctx, lane, width);
    rw.insert_op(ctx, &half_op);
    let (in_row, half) = (in_row.get_result(ctx), half_op.get_result(ctx));

    let along = match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => match generation {
            WmmaGeneration::Rdna3 => i,
            WmmaGeneration::Rdna4 => {
                let per_half = insert_i32_const(ctx, rw, matrix.shape.k as i32 / 2);
                let offset = mul(ctx, rw, half, per_half);
                add(ctx, rw, i, offset)
            }
        },
        MatrixIdent::Accumulator => match generation {
            WmmaGeneration::Rdna3 => {
                let two = insert_i32_const(ctx, rw, 2);
                let row = mul(ctx, rw, i, two);
                add(ctx, rw, row, half)
            }
            WmmaGeneration::Rdna4 => {
                let block = insert_i32_const(ctx, rw, ACCUMULATOR_REGISTERS as i32);
                let offset = mul(ctx, rw, half, block);
                add(ctx, rw, i, offset)
            }
        },
    };

    // A is indexed by its row and walks `k`; B and the accumulator are the other way round.
    let lane_names_row = matrix.ident == MatrixIdent::A;
    match (axis, lane_names_row) {
        (Axis::Row, true) | (Axis::Col, false) => in_row,
        (Axis::Row, false) | (Axis::Col, true) => along,
    }
}

/// Lowers `row_index` and `col_index`, which differ only in the axis they ask for.
macro_rules! lower_axis_index {
    ($cube_op:ty, $axis:expr) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rw: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let old_op = self.get_operation();
                let lane = self.lane_id(ctx);
                let i = self.i(ctx);
                let handle = self.matrix_ty(ctx).clone();
                let matrix = *handle.deref(ctx);

                let index = axis_index(ctx, rw, &matrix, $axis, lane, i);
                rw.replace_operation_with_values(ctx, old_op, vec![index]);
                Ok(())
            }
        }
    };
}

lower_axis_index!(RowIndexOp, Axis::Row);
lower_axis_index!(ColIndexOp, Axis::Col);

/// The LLVM vector holding the registers `value` points at.
///
/// The manual ops carry their operands as arrays rather than fragments, but the registers are
/// the same ones, so the array is read as the vector the instruction expects.
fn registers_as_vector(
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
fn registers_value(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    vector_ty: TypeHandle,
) -> Value {
    let ty = value.get_type(ctx);
    if ty.deref(ctx).is::<LlvmPointerType>() {
        return load_fragment(ctx, rw, value, vector_ty);
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
    rw.insert_op(ctx, &poison);
    let mut acc = poison.get_result(ctx);

    for register in 0..count {
        let element = llvm::ExtractValueOp::new(ctx, value, vec![register as u32])
            .expect("a constant index into the register array");
        rw.insert_op(ctx, &element);
        let element = element.get_result(ctx);

        for lane in 0..per_register {
            let value = if per_register == 1 {
                element
            } else {
                let from = insert_i32_const(ctx, rw, lane as i32);
                let extract = llvm::ExtractElementOp::new(ctx, element, from);
                rw.insert_op(ctx, &extract);
                extract.get_result(ctx)
            };
            let to = insert_i32_const(ctx, rw, (register * per_register + lane) as i32);
            let insert = llvm::InsertElementOp::new(ctx, acc, value, to);
            rw.insert_op(ctx, &insert);
            acc = insert.get_result(ctx);
        }
    }
    acc
}

#[op_interface_impl]
impl ToLLVMDialect for MmaManualOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let (a, b, c, d) = (
            self.registers_a(ctx),
            self.registers_b(ctx),
            self.registers_c(ctx),
            self.registers_d(ctx),
        );

        let (ab_ty, ab_elem) = registers_as_vector(ctx, operands_info, a);
        let (cd_ty, cd_elem) = registers_as_vector(ctx, operands_info, c);

        let a_val = registers_value(ctx, rw, a, ab_ty);
        let b_val = registers_value(ctx, rw, b, ab_ty);
        let c_val = registers_value(ctx, rw, c, cd_ty);

        let MatrixShape { m, n, k } = *self.shape(ctx).clone();
        let ab = wmma_format(ctx, ab_elem);
        let cd = wmma_format(ctx, cd_elem);
        let name = format!(
            "llvm.amdgcn.wmma.{cd}.{m}x{n}x{k}.{ab}.{}.{}",
            llvm_mangled_ty(ctx, cd_ty),
            llvm_mangled_ty(ctx, ab_ty),
        );

        let mut args = vec![a_val, b_val, c_val];
        let mut arg_tys = vec![ab_ty, ab_ty, cd_ty];
        if ctx.wmma().pads_half_accumulator() && is_half(ctx, cd_elem) {
            let low_half = insert_bool_const(ctx, rw, false);
            arg_tys.push(low_half.get_type(ctx));
            args.push(low_half);
        }

        let fn_ty = FuncType::get(ctx, cd_ty, arg_tys, false);
        let op = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
        rw.insert_op(ctx, &op);
        store_fragment(ctx, rw, d, op.get_result(ctx));

        rw.erase_operation(ctx, old_op);
        Ok(())
    }
}
