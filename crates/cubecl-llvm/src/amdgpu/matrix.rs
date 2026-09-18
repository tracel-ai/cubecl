//! AMDGPU matrix operations.

use crate::{
    amdgpu::plane::lane_id,
    prelude::*,
    shared::matrix::{registers_as_vector, registers_value},
};
use cubecl_core::ir::{
    amd::AmdWmma,
    dialect::matrix::{
        CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp,
        RowIndexOp, StMatrixOp, StoreOp,
    },
    types::{MatrixIdent, MatrixLayout, MatrixShape, matrix::MatrixType},
};

#[derive(Debug, Error)]
#[error(
    "casting a {0} fragment to a {1} one needs a cross-lane relayout the AMDGPU lowering does \
     not implement: an accumulator spreads several rows over each lane, an A or B fragment one"
)]
pub struct MatrixRelayoutUnsupported(MatrixIdent, MatrixIdent);

#[derive(Debug, Error)]
#[error("a k of {0} does not divide into {1}-deep WMMA instructions")]
pub struct MatrixDepthUnsupported(usize, usize);

#[derive(Debug, Error)]
#[error("no WMMA instruction takes a {0} fragment element")]
pub struct MatrixElemUnsupported(String);

/// RDNA3 pads 16-bit accumulators to 32-bit registers.
fn pads_half_accumulator(generation: AmdWmma) -> bool {
    generation == AmdWmma::Rdna3
}

/// Tiles must contain a whole number of WMMA instructions.
fn instruction_steps(generation: AmdWmma, k: usize) -> Option<(usize, usize)> {
    let instruction_k = match generation {
        AmdWmma::Rdna3 => 16.min(k),
        AmdWmma::Rdna4 => k,
    };
    (instruction_k > 0 && k.is_multiple_of(instruction_k))
        .then(|| (instruction_k, k / instruction_k))
}

impl CtxWmma for Context {}

pub trait CtxWmma: ContextExt {
    fn wmma(&self) -> AmdWmma {
        *self
            .aux_ty::<Option<AmdWmma>>()
            .as_ref()
            .expect("matrix ops are only compiled for devices that have WMMA")
    }
    fn set_wmma(&mut self, generation: Option<AmdWmma>) {
        self.set_aux_ty(generation);
    }
}

fn is_half(ctx: &Context, elem: TypeHandle) -> bool {
    elem.is_float16(ctx) || elem.is_bfloat16(ctx)
}

const ACCUMULATOR_REGISTERS: usize = 8;

const LANES_PER_ROW: u32 = 16;

/// Fragment size and element spacing per lane.
fn fragment_layout(ctx: &Context, matrix: &MatrixType) -> (usize, usize) {
    let generation = ctx.wmma();
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => (generation.frag_ab_elems(matrix.shape.k), 1),
        MatrixIdent::Accumulator => {
            let padded = is_half(ctx, matrix.elem_ty) && pads_half_accumulator(generation);
            (ACCUMULATOR_REGISTERS, if padded { 2 } else { 1 })
        }
    }
}

pub(crate) fn fragment_ty(ctx: &Context, matrix: &MatrixType) -> TypeHandle {
    let (elems, step) = fragment_layout(ctx, matrix);
    let elem = cube_type_to_llvm(ctx, matrix.elem_ty);
    LlvmVectorType::get(ctx, elem, (elems * step) as u32, VectorTypeKind::Fixed).into()
}

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

fn mul(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

fn add(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

#[derive(Clone, Copy)]
struct LanePosition {
    in_row: Value,
    half: Value,
}

impl LanePosition {
    fn of(ctx: &mut Context, rw: &mut DialectConversionRewriter, lane: Value) -> Self {
        let width = insert_i32_const(ctx, rw, LANES_PER_ROW as i32);

        let in_row = llvm::URemOp::new(ctx, lane, width);
        let half = llvm::UDivOp::new(ctx, lane, width);

        LanePosition {
            in_row: insert(ctx, rw, &in_row),
            half: insert(ctx, rw, &half),
        }
    }

    fn current(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Self {
        let lane = lane_id(ctx, rw);
        Self::of(ctx, rw, lane)
    }
}

/// Shared fragment coordinates for memory access and matrix indexing.
fn along(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    i: Value,
    lane: LanePosition,
) -> Value {
    let half = lane.half;
    match (matrix.ident, ctx.wmma()) {
        (MatrixIdent::A | MatrixIdent::B, AmdWmma::Rdna3) => i,
        (MatrixIdent::A | MatrixIdent::B, AmdWmma::Rdna4) => {
            let per_half = insert_i32_const(ctx, rw, matrix.shape.k as i32 / 2);
            let offset = mul(ctx, rw, half, per_half);
            add(ctx, rw, i, offset)
        }
        (MatrixIdent::Accumulator, AmdWmma::Rdna3) => {
            let two = insert_i32_const(ctx, rw, 2);
            let row = mul(ctx, rw, i, two);
            add(ctx, rw, row, half)
        }
        (MatrixIdent::Accumulator, AmdWmma::Rdna4) => {
            let block = insert_i32_const(ctx, rw, ACCUMULATOR_REGISTERS as i32);
            let offset = mul(ctx, rw, half, block);
            add(ctx, rw, i, offset)
        }
    }
}

fn element_index(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    layout: MatrixLayout,
    i: usize,
    lane: LanePosition,
    stride: Value,
) -> Value {
    let step = insert_i32_const(ctx, rw, i as i32);
    let along = along(ctx, rw, matrix, step, lane);
    let across = lane.in_row;

    if is_strided(matrix.ident, layout) {
        let scaled = mul(ctx, rw, along, stride);
        add(ctx, rw, scaled, across)
    } else {
        let scaled = mul(ctx, rw, across, stride);
        add(ctx, rw, along, scaled)
    }
}

fn is_strided(ident: MatrixIdent, layout: MatrixLayout) -> bool {
    matches!(
        (ident, layout),
        (MatrixIdent::A, MatrixLayout::ColMajor)
            | (MatrixIdent::B, MatrixLayout::RowMajor)
            | (MatrixIdent::Accumulator, MatrixLayout::RowMajor)
    )
}

/// Vector access requires consecutive elements in both memory and registers.
fn fragment_is_contiguous(
    generation: AmdWmma,
    matrix: &MatrixType,
    layout: MatrixLayout,
    step: usize,
) -> bool {
    if step != 1 || is_strided(matrix.ident, layout) {
        return false;
    }
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => true,
        MatrixIdent::Accumulator => generation == AmdWmma::Rdna4,
    }
}

fn element_ptr(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    base: Value,
    index: Value,
    elem_ty: TypeHandle,
) -> Value {
    let gep = llvm::GetElementPtrOp::new(ctx, base, vec![llvm::GepIndex::Value(index)], elem_ty);
    insert(ctx, rw, &gep)
}

fn load_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::LoadOp::new(ctx, matrix, ty);
    insert(ctx, rw, &op)
}

fn store_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    value: Value,
) {
    let op = llvm::StoreOp::new(ctx, value, matrix);
    rw.insert_op(ctx, &op);
}

/// Tile accesses guarantee only element alignment; row strides may include padding.
fn load_tile(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    ptr: Value,
    ty: TypeHandle,
    align: u32,
) -> Value {
    let op = llvm::LoadOp::new(ctx, ptr, ty);
    op.set_alignment(ctx, align);
    insert(ctx, rw, &op)
}

fn store_tile(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    ptr: Value,
    value: Value,
    align: u32,
) {
    let op = llvm::StoreOp::new(ctx, value, ptr);
    op.set_alignment(ctx, align);
    rw.insert_op(ctx, &op);
}

pub(crate) fn fill(
    op: &FillOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let matrix = op.matrix(ctx);
    let value = op.value(ctx);

    let ty = matrix_of(ctx, operands_info, matrix);
    let (elems, step) = fragment_layout(ctx, &ty);
    let frag_ty = fragment_ty(ctx, &ty);
    let lanes = elems * step;

    let filled = insert_splat(ctx, rw, frag_ty, value, lanes);
    store_fragment(ctx, rw, matrix, filled);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

pub(crate) fn load(
    op: &LoadOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let matrix = op.matrix(ctx);
    let source = op.source(ctx);
    let stride = op.stride(ctx);
    let layout = op.layout(ctx).0;

    let ty = matrix_of(ctx, operands_info, matrix);
    let (elems, step) = fragment_layout(ctx, &ty);
    let frag_ty = fragment_ty(ctx, &ty);
    let elem_ty = cube_type_to_llvm(ctx, ty.elem_ty);
    let align = scalar_alignment(ctx, ty.elem_ty);
    let lane = LanePosition::current(ctx, rw);

    if fragment_is_contiguous(ctx.wmma(), &ty, layout, step) {
        let index = element_index(ctx, rw, &ty, layout, 0, lane, stride);
        let ptr = element_ptr(ctx, rw, source, index, elem_ty);
        let frag = load_tile(ctx, rw, ptr, frag_ty, align);
        store_fragment(ctx, rw, matrix, frag);

        rw.erase_operation(ctx, old_op);
        return Ok(());
    }

    let poison = llvm::PoisonOp::new(ctx, frag_ty);
    let mut frag = insert(ctx, rw, &poison);

    for i in 0..elems {
        let index = element_index(ctx, rw, &ty, layout, i, lane, stride);
        let ptr = element_ptr(ctx, rw, source, index, elem_ty);
        let value = load_tile(ctx, rw, ptr, elem_ty, align);

        let slot = insert_i32_const(ctx, rw, (i * step) as i32);
        let op = llvm::InsertElementOp::new(ctx, frag, value, slot);
        frag = insert(ctx, rw, &op);
    }

    store_fragment(ctx, rw, matrix, frag);
    rw.erase_operation(ctx, old_op);
    Ok(())
}

pub(crate) fn store(
    op: &StoreOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let matrix = op.matrix(ctx);
    let destination = op.destination(ctx);
    let stride = op.stride(ctx);
    let layout = op.layout(ctx).0;

    let ty = matrix_of(ctx, operands_info, matrix);
    let (elems, step) = fragment_layout(ctx, &ty);
    let frag_ty = fragment_ty(ctx, &ty);
    let elem_ty = cube_type_to_llvm(ctx, ty.elem_ty);
    let align = scalar_alignment(ctx, ty.elem_ty);
    let lane = LanePosition::current(ctx, rw);

    let frag = load_fragment(ctx, rw, matrix, frag_ty);

    if fragment_is_contiguous(ctx.wmma(), &ty, layout, step) {
        let index = element_index(ctx, rw, &ty, layout, 0, lane, stride);
        let ptr = element_ptr(ctx, rw, destination, index, elem_ty);
        store_tile(ctx, rw, ptr, frag, align);

        rw.erase_operation(ctx, old_op);
        return Ok(());
    }

    for i in 0..elems {
        let slot = insert_i32_const(ctx, rw, (i * step) as i32);
        let extract = llvm::ExtractElementOp::new(ctx, frag, slot);
        let element = insert(ctx, rw, &extract);

        let index = element_index(ctx, rw, &ty, layout, i, lane, stride);
        let ptr = element_ptr(ctx, rw, destination, index, elem_ty);
        store_tile(ctx, rw, ptr, element, align);
    }

    rw.erase_operation(ctx, old_op);
    Ok(())
}

fn instruction_k(ctx: &Context) -> usize {
    match ctx.wmma() {
        AmdWmma::Rdna3 => 16,
        AmdWmma::Rdna4 => 32,
    }
}

fn unsupported_elem(ctx: &Context, ab: TypeHandle, cd: TypeHandle) -> MatrixElemUnsupported {
    let culprit = if wmma_format(ctx, ab).is_none() {
        ab
    } else {
        cd
    };
    MatrixElemUnsupported(culprit.disp(ctx).to_string())
}

fn wmma_format(ctx: &Context, elem: TypeHandle) -> Option<&'static str> {
    if elem.is_float32(ctx) {
        Some("f32")
    } else if elem.is_bfloat16(ctx) {
        Some("bf16")
    } else if elem.is_float16(ctx) {
        Some("f16")
    } else {
        None
    }
}

fn fragment_slice(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    step: usize,
    width: usize,
) -> Value {
    shuffle(
        ctx,
        rw,
        fragment,
        (0..width).map(|i| (step * width + i) as i32).collect(),
    )
}

fn shuffle(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    mask: Vec<i32>,
) -> Value {
    let op = llvm::ShuffleVectorOp::new(ctx, fragment, fragment, mask);
    insert(ctx, rw, &op)
}

struct WmmaCall {
    /// Tile dimensions.
    shape: MatrixShape,
    /// A/B element format.
    ab: &'static str,
    /// C/D element format.
    cd: &'static str,
    /// Whether the accumulator uses 16-bit elements.
    cd_is_half: bool,
}

fn emit_wmma(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    call: WmmaCall,
    (a_val, b_val, c_val): (Value, Value, Value),
    ab_ty: TypeHandle,
    cd_ty: TypeHandle,
) -> Option<Value> {
    let WmmaCall {
        shape: MatrixShape { m, n, k },
        ab,
        cd,
        cd_is_half,
    } = call;
    let generation = ctx.wmma();
    let (instruction_k, steps) = instruction_steps(generation, k)?;
    let pads_half = pads_half_accumulator(generation) && cd_is_half;

    let mut acc = c_val;
    for step in 0..steps {
        let (a_arg, b_arg, arg_ty) = if steps == 1 {
            (a_val, b_val, ab_ty)
        } else {
            let a_slice = fragment_slice(ctx, rw, a_val, step, instruction_k);
            let b_slice = fragment_slice(ctx, rw, b_val, step, instruction_k);
            let ty = a_slice.get_type(ctx);
            (a_slice, b_slice, ty)
        };

        let name = format!(
            "llvm.amdgcn.wmma.{cd}.{m}x{n}x{instruction_k}.{ab}.{}.{}",
            llvm_mangled_ty(ctx, cd_ty),
            llvm_mangled_ty(ctx, arg_ty),
        );

        let mut args = vec![a_arg, b_arg, acc];
        let mut arg_tys = vec![arg_ty, arg_ty, cd_ty];
        // RDNA3 selects a register half with `opsel`; RDNA4 uses packed accumulators.
        if pads_half {
            let low_half = insert_bool_const(ctx, rw, false);
            arg_tys.push(low_half.get_type(ctx));
            args.push(low_half);
        }

        let fn_ty = FuncType::get(ctx, cd_ty, arg_tys, false);
        let op = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
        acc = insert(ctx, rw, &op);
    }
    Some(acc)
}

pub(crate) fn multiply_accumulate(
    op: &MultiplyAccumulateOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let (a, b, c, d) = (op.mat_a(ctx), op.mat_b(ctx), op.mat_c(ctx), op.mat_d(ctx));

    let a_ty = matrix_of(ctx, operands_info, a);
    let c_ty = matrix_of(ctx, operands_info, c);
    let ab_frag_ty = fragment_ty(ctx, &a_ty);
    let cd_frag_ty = fragment_ty(ctx, &c_ty);

    let a_val = load_fragment(ctx, rw, a, ab_frag_ty);
    let b_val = load_fragment(ctx, rw, b, ab_frag_ty);
    let c_val = load_fragment(ctx, rw, c, cd_frag_ty);

    let (Some(ab), Some(cd)) = (
        wmma_format(ctx, a_ty.elem_ty),
        wmma_format(ctx, c_ty.elem_ty),
    ) else {
        return input_err!(
            op.loc(ctx),
            unsupported_elem(ctx, a_ty.elem_ty, c_ty.elem_ty)
        );
    };
    let k = a_ty.shape.k;
    let call = WmmaCall {
        shape: a_ty.shape,
        ab,
        cd,
        cd_is_half: is_half(ctx, c_ty.elem_ty),
    };
    let Some(result) = emit_wmma(ctx, rw, call, (a_val, b_val, c_val), ab_frag_ty, cd_frag_ty)
    else {
        return input_err!(op.loc(ctx), MatrixDepthUnsupported(k, instruction_k(ctx)));
    };
    store_fragment(ctx, rw, d, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

pub(crate) fn cast(
    op: &CastOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let input = op.input(ctx);
    let output = op.output(ctx);

    let in_ty = matrix_of(ctx, operands_info, input);
    let out_ty = matrix_of(ctx, operands_info, output);
    let in_frag_ty = fragment_ty(ctx, &in_ty);

    let value = load_fragment(ctx, rw, input, in_frag_ty);

    let (elems, in_step) = fragment_layout(ctx, &in_ty);
    let (out_elems, out_step) = fragment_layout(ctx, &out_ty);

    // Casts between input and accumulator fragments require a different lane layout.
    if (in_ty.ident == MatrixIdent::Accumulator) != (out_ty.ident == MatrixIdent::Accumulator) {
        return input_err!(
            op.loc(ctx),
            MatrixRelayoutUnsupported(in_ty.ident, out_ty.ident)
        );
    }
    assert_eq!(
        elems, out_elems,
        "a cast keeps the element count and changes only their width"
    );

    let dense = if in_step == 1 {
        value
    } else {
        shuffle(
            ctx,
            rw,
            value,
            (0..elems).map(|i| (i * in_step) as i32).collect(),
        )
    };

    let in_bits = in_ty.elem_ty.size_bits(ctx);
    let out_bits = out_ty.elem_ty.size_bits(ctx);
    let dense_out_ty: TypeHandle = LlvmVectorType::get(
        ctx,
        cube_type_to_llvm(ctx, out_ty.elem_ty),
        elems as u32,
        VectorTypeKind::Fixed,
    )
    .into();
    let cast = if in_bits > out_bits {
        fptrunc(ctx, rw, dense, dense_out_ty)
    } else if in_bits < out_bits {
        fpext(ctx, rw, dense, dense_out_ty)
    } else if in_ty.elem_ty == out_ty.elem_ty {
        dense
    } else if is_half(ctx, in_ty.elem_ty) && is_half(ctx, out_ty.elem_ty) {
        // Conversions between f16 and bf16 require an f32 intermediate.
        let wide_ty: TypeHandle = LlvmVectorType::get(
            ctx,
            FP32Type::get(ctx).into(),
            elems as u32,
            VectorTypeKind::Fixed,
        )
        .into();
        let wide = fpext(ctx, rw, dense, wide_ty);
        fptrunc(ctx, rw, wide, dense_out_ty)
    } else {
        debug_assert_eq!(
            cube_type_to_llvm(ctx, in_ty.elem_ty),
            cube_type_to_llvm(ctx, out_ty.elem_ty),
            "a cast of the same width between two distinct LLVM types needs a conversion, \
             and neither `fpext` nor `fptrunc` is one"
        );
        dense
    };

    let result = if out_step == 1 {
        cast
    } else {
        shuffle(
            ctx,
            rw,
            cast,
            (0..elems * out_step)
                .map(|i| (i / out_step) as i32)
                .collect(),
        )
    };
    store_fragment(ctx, rw, output, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

fn fpext(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::FPExtOp::new(ctx, value, ty);
    op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
    insert(ctx, rw, &op)
}

fn fptrunc(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::FPTruncOp::new(ctx, value, ty);
    op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
    insert(ctx, rw, &op)
}

enum Axis {
    Row,
    Col,
}

fn axis_index(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    axis: Axis,
    lane: Value,
    i: Value,
) -> Value {
    let lane = LanePosition::of(ctx, rw, lane);
    let along = along(ctx, rw, matrix, i, lane);

    let lane_names_row = matrix.ident == MatrixIdent::A;
    match (axis, lane_names_row) {
        (Axis::Row, true) | (Axis::Col, false) => lane.in_row,
        (Axis::Row, false) | (Axis::Col, true) => along,
    }
}

macro_rules! lower_axis_index {
    ($fn_name:ident, $cube_op:ty, $axis:expr) => {
        pub(crate) fn $fn_name(
            op: &$cube_op,
            ctx: &mut Context,
            rw: &mut DialectConversionRewriter,
            _operands_info: &OperandsInfo,
        ) -> Result<()> {
            let old_op = op.get_operation();
            let lane = op.lane_id(ctx);
            let i = op.i(ctx);
            let handle = op.matrix_ty(ctx).clone();
            let matrix = *handle.deref(ctx);

            let index = axis_index(ctx, rw, &matrix, $axis, lane, i);
            rw.replace_operation_with_values(ctx, old_op, vec![index]);
            Ok(())
        }
    };
}

lower_axis_index!(row_index, RowIndexOp, Axis::Row);
lower_axis_index!(col_index, ColIndexOp, Axis::Col);

pub(crate) fn mma_manual(
    op: &MmaManualOp,
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

    let (ab_ty, ab_elem) = registers_as_vector(ctx, operands_info, a);
    let (cd_ty, cd_elem) = registers_as_vector(ctx, operands_info, c);

    let a_val = registers_value(ctx, rw, a, ab_ty);
    let b_val = registers_value(ctx, rw, b, ab_ty);
    let c_val = registers_value(ctx, rw, c, cd_ty);

    let (Some(ab), Some(cd)) = (wmma_format(ctx, ab_elem), wmma_format(ctx, cd_elem)) else {
        return input_err!(op.loc(ctx), unsupported_elem(ctx, ab_elem, cd_elem));
    };
    let shape = *op.shape(ctx).clone();
    let call = WmmaCall {
        shape,
        ab,
        cd,
        cd_is_half: is_half(ctx, cd_elem),
    };
    let Some(result) = emit_wmma(ctx, rw, call, (a_val, b_val, c_val), ab_ty, cd_ty) else {
        return input_err!(
            op.loc(ctx),
            MatrixDepthUnsupported(shape.k, instruction_k(ctx))
        );
    };
    store_fragment(ctx, rw, d, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::ir::types::MatrixScope;
    use cubecl_core::ir::types::scalar::{Float16Type, Float32Type};

    fn matrix(ident: MatrixIdent, elem_ty: TypeHandle, k: usize) -> MatrixType {
        MatrixType {
            ident,
            shape: MatrixShape { m: 16, n: 16, k },
            elem_ty,
            layout: MatrixLayout::RowMajor,
            scope: MatrixScope::Plane,
        }
    }

    fn f16(ctx: &mut Context) -> TypeHandle {
        Float16Type::get(ctx).into()
    }

    fn f32(ctx: &mut Context) -> TypeHandle {
        Float32Type::get(ctx).into()
    }

    #[test]
    fn each_generation_holds_its_own_share_of_k() {
        let mut ctx = Context::default();
        let f16 = f16(&mut ctx);

        for (generation, elems) in [(AmdWmma::Rdna3, 16), (AmdWmma::Rdna4, 8)] {
            ctx.set_wmma(Some(generation));
            let a = matrix(MatrixIdent::A, f16, 16);
            assert_eq!(fragment_layout(&ctx, &a), (elems, 1), "{generation:?}");
        }
    }

    #[test]
    fn only_a_half_accumulator_on_rdna3_is_padded() {
        let mut ctx = Context::default();
        let (f16, f32) = (f16(&mut ctx), f32(&mut ctx));

        for (generation, elem, step) in [
            (AmdWmma::Rdna3, f16, 2),
            (AmdWmma::Rdna3, f32, 1),
            (AmdWmma::Rdna4, f16, 1),
            (AmdWmma::Rdna4, f32, 1),
        ] {
            ctx.set_wmma(Some(generation));
            let acc = matrix(MatrixIdent::Accumulator, elem, 16);
            assert_eq!(
                fragment_layout(&ctx, &acc),
                (ACCUMULATOR_REGISTERS, step),
                "{generation:?}"
            );
        }
    }

    #[test]
    fn a_tile_deeper_than_the_instruction_is_several_of_them() {
        assert_eq!(instruction_steps(AmdWmma::Rdna3, 16), Some((16, 1)));
        assert_eq!(instruction_steps(AmdWmma::Rdna3, 32), Some((16, 2)));
        assert_eq!(instruction_steps(AmdWmma::Rdna4, 32), Some((32, 1)));
    }

    #[test]
    fn a_tile_that_does_not_divide_has_no_lowering() {
        assert_eq!(instruction_steps(AmdWmma::Rdna3, 24), None);
        assert_eq!(instruction_steps(AmdWmma::Rdna4, 0), None);
    }

    #[test]
    fn the_layout_decides_which_axis_carries_the_stride() {
        use MatrixIdent::{A, Accumulator, B};
        use MatrixLayout::{ColMajor, RowMajor};

        assert!(is_strided(A, ColMajor) && !is_strided(A, RowMajor));
        assert!(is_strided(B, RowMajor) && !is_strided(B, ColMajor));
        assert!(is_strided(Accumulator, RowMajor) && !is_strided(Accumulator, ColMajor));
    }

    #[test]
    fn only_a_fragment_dense_on_both_sides_loads_as_one_access() {
        let mut ctx = Context::default();
        let (f16, f32) = (f16(&mut ctx), f32(&mut ctx));
        let a = matrix(MatrixIdent::A, f16, 16);
        let acc16 = matrix(MatrixIdent::Accumulator, f16, 16);
        let acc32 = matrix(MatrixIdent::Accumulator, f32, 16);

        assert!(fragment_is_contiguous(
            AmdWmma::Rdna3,
            &a,
            MatrixLayout::RowMajor,
            1
        ));
        assert!(!fragment_is_contiguous(
            AmdWmma::Rdna3,
            &a,
            MatrixLayout::ColMajor,
            1
        ));

        assert!(!fragment_is_contiguous(
            AmdWmma::Rdna3,
            &acc32,
            MatrixLayout::ColMajor,
            1
        ));
        assert!(fragment_is_contiguous(
            AmdWmma::Rdna4,
            &acc32,
            MatrixLayout::ColMajor,
            1
        ));
        assert!(!fragment_is_contiguous(
            AmdWmma::Rdna4,
            &acc16,
            MatrixLayout::ColMajor,
            2
        ));
    }
}

#[derive(Debug, Error)]
#[error(
    "`{0}` is `ldmatrix`/`stmatrix`, an NVIDIA instruction; the AMDGPU backend does not \
     advertise it and has nothing to lower it to"
)]
pub struct MatrixTileMoveUnsupported(&'static str);

macro_rules! unsupported_tile_move {
    ($fn_name:ident, $cube_op:ty) => {
        pub(crate) fn $fn_name(
            op: &$cube_op,
            ctx: &mut Context,
            _rw: &mut DialectConversionRewriter,
            _operands_info: &OperandsInfo,
        ) -> Result<()> {
            input_err!(op.loc(ctx), MatrixTileMoveUnsupported(stringify!($cube_op)))
        }
    };
}

unsupported_tile_move!(ld_matrix, LdMatrixOp);
unsupported_tile_move!(st_matrix, StMatrixOp);
