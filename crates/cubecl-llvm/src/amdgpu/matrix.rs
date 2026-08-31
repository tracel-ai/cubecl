//! Lowering of the matrix operations to the wavefront's WMMA instructions.
//!
//! A fragment is a vector held across the lanes of the wavefront: A and B hold the `k` range of
//! one row or column, the accumulator eight rows' worth. The layouts differ by generation —
//! RDNA3 gives both halves of the wave the whole `k` range and pads 16 bit accumulators out to
//! 32 bits per element, RDNA4 splits `k` between the halves and packs densely — so the element
//! counts and the index arithmetic are derived from [`AmdWmma`] rather than fixed.

use cubecl_core::ir::ContextExt;
use cubecl_core::ir::amd::AmdWmma;
use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use cubecl_core::ir::types::{MatrixIdent, MatrixLayout, MatrixShape};

use pliron::input_err;
use pliron::printable::Printable;
use thiserror::Error;

use crate::amdgpu::plane::lane_id;
use crate::shared::to_llvm::prelude::*;
use crate::shared::to_llvm::ty::scalar_alignment;

/// A cast that would have to move elements between lanes, which this lowering cannot do.
#[derive(Debug, Error)]
#[error(
    "casting a {0} fragment to a {1} one needs a cross-lane relayout the AMDGPU lowering does \
     not implement: an accumulator spreads several rows over each lane, an A or B fragment one"
)]
pub struct MatrixRelayoutUnsupported(MatrixIdent, MatrixIdent);

/// A tile whose depth is not a whole number of hardware instructions.
#[derive(Debug, Error)]
#[error("a k of {0} does not divide into {1}-deep WMMA instructions")]
pub struct MatrixDepthUnsupported(usize, usize);

/// A fragment element type no WMMA instruction takes.
#[derive(Debug, Error)]
#[error("no WMMA instruction takes a {0} fragment element")]
pub struct MatrixElemUnsupported(String);

/// Whether a 16 bit accumulator is padded to one element per 32 bit register.
fn pads_half_accumulator(generation: AmdWmma) -> bool {
    generation == AmdWmma::Rdna3
}

/// How a tile of depth `k` splits into hardware instructions: the `k` of one instruction,
/// and how many of them the tile takes.
///
/// gfx11 only has the `16x16x16` WMMA shapes, but rocWMMA advertises `16x16x32` on it as
/// well and implements it as two instructions, so the device properties offer that tile and
/// callers take it up. A tile deeper than the instruction is therefore not an error: it is
/// several instructions over consecutive slices of the `k` range, chained through the
/// accumulator. gfx12 has the wider instruction and always runs in one step.
fn instruction_steps(generation: AmdWmma, k: usize) -> Option<(usize, usize)> {
    let instruction_k = match generation {
        AmdWmma::Rdna3 => 16.min(k),
        AmdWmma::Rdna4 => k,
    };
    (instruction_k > 0 && k.is_multiple_of(instruction_k))
        .then(|| (instruction_k, k / instruction_k))
}

impl CtxWmma for Context {}

/// The WMMA generation on the context.
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
        MatrixIdent::A | MatrixIdent::B => (generation.frag_ab_elems(matrix.shape.k), 1),
        MatrixIdent::Accumulator => {
            let padded = is_half(ctx, matrix.elem_ty) && pads_half_accumulator(generation);
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
    insert(ctx, rw, &op)
}

/// `lhs + rhs` over `i32`.
fn add(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

/// Where in the fragment's row or column a lane sits, and which half of the wave it is in.
#[derive(Clone, Copy)]
struct LanePosition {
    in_row: Value,
    half: Value,
}

impl LanePosition {
    /// Split `lane`, an index within the wavefront, into the two halves the fragment
    /// layouts are written in terms of.
    fn of(ctx: &mut Context, rw: &mut DialectConversionRewriter, lane: Value) -> Self {
        let width = insert_i32_const(ctx, rw, LANES_PER_ROW as i32);

        let in_row = llvm::URemOp::new(ctx, lane, width);
        let half = llvm::UDivOp::new(ctx, lane, width);

        LanePosition {
            in_row: insert(ctx, rw, &in_row),
            half: insert(ctx, rw, &half),
        }
    }

    /// This lane's position within its own wavefront.
    fn current(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Self {
        let lane = lane_id(ctx, rw);
        Self::of(ctx, rw, lane)
    }
}

/// How far along the fragment's own axis this lane's `i`th element sits: the `k` range for
/// A and B, the rows of the output for the accumulator.
///
/// This is the only place the fragment layout is written down. [`element_index`] walks it
/// with a constant `i` to place a load or a store, and [`axis_index`] with a runtime one to
/// answer `row_index` and `col_index`. A second copy that drifted would have a kernel
/// addressing its own fragment at coordinates the loads never wrote.
fn along(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    i: Value,
    lane: LanePosition,
) -> Value {
    let half = lane.half;
    match (matrix.ident, ctx.wmma()) {
        // Every lane holds the whole `k` range, duplicated across the halves of the wave.
        (MatrixIdent::A | MatrixIdent::B, AmdWmma::Rdna3) => i,
        // Each half holds its own part of the `k` range.
        (MatrixIdent::A | MatrixIdent::B, AmdWmma::Rdna4) => {
            let per_half = insert_i32_const(ctx, rw, matrix.shape.k as i32 / 2);
            let offset = mul(ctx, rw, half, per_half);
            add(ctx, rw, i, offset)
        }
        // The halves interleave row by row.
        (MatrixIdent::Accumulator, AmdWmma::Rdna3) => {
            let two = insert_i32_const(ctx, rw, 2);
            let row = mul(ctx, rw, i, two);
            add(ctx, rw, row, half)
        }
        // Each half gets a contiguous block of eight rows.
        (MatrixIdent::Accumulator, AmdWmma::Rdna4) => {
            let block = insert_i32_const(ctx, rw, ACCUMULATOR_REGISTERS as i32);
            let offset = mul(ctx, rw, half, block);
            add(ctx, rw, i, offset)
        }
    }
}

/// The index into the backing memory of this lane's `i`th element of `matrix`.
///
/// The element sits at [`along`] on the fragment's own axis and at the lane's place in the
/// row on the other; `layout` decides which of the two the stride multiplies.
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

/// Whether the layout puts `along` on the stride rather than `across`.
///
/// Whichever of the two the layout makes contiguous gets the stride.
fn is_strided(ident: MatrixIdent, layout: MatrixLayout) -> bool {
    matches!(
        (ident, layout),
        (MatrixIdent::A, MatrixLayout::ColMajor)
            | (MatrixIdent::B, MatrixLayout::RowMajor)
            | (MatrixIdent::Accumulator, MatrixLayout::RowMajor)
    )
}

/// Whether a fragment's elements sit one after another in memory, so the whole
/// fragment is one vector access instead of `elems` scalar ones.
///
/// Two things have to hold. The layout must leave `along` unstrided, which makes
/// [`element_index`] `along + across * stride` and so a step of one per `i`. And
/// `along` itself must advance by one per `i`: it does for A and B, and for an
/// RDNA4 accumulator, but an RDNA3 accumulator steps two rows at a time because
/// the halves of the wave interleave. `step` covers the matching gap on the
/// register side -- a padded 16 bit accumulator spreads its elements over every
/// second slot, which no contiguous load can fill.
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

/// The address of element `index` of `base`, whose elements are `elem_ty`.
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

/// Loads the fragment `matrix` points at.
fn load_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    ty: TypeHandle,
) -> Value {
    let op = llvm::LoadOp::new(ctx, matrix, ty);
    insert(ctx, rw, &op)
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

/// Loads `ty` out of the tile `ptr` points into, at the alignment the tile guarantees.
///
/// A fragment sits in an `alloca` of the vector itself, so an access to it is aligned to the
/// whole vector by construction. A tile in memory is not: its rows are `stride` elements
/// apart and a caller is free to pad that stride, so an access can land on any element
/// boundary. Left implicit, LLVM assumes the ABI alignment of the type it is given -- 64
/// bytes for a `<32 x half>` A fragment -- and a wide `ds_read` on an address that does not
/// meet it reads the wrong data. The element's alignment is what the tile actually promises,
/// so it is what is asked for; LLVM raises it on its own wherever it can prove more.
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

/// Stores `value` into the tile `ptr` points into. The counterpart of [`load_tile`], and
/// aligned for the same reason.
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
        let align = scalar_alignment(ctx, ty.elem_ty);
        let lane = LanePosition::current(ctx, rw);

        // One vector load where the elements are consecutive, at the element's alignment
        // like the element-wise path below. The base of a tile is aligned far past that --
        // an LDS block to 128 bytes -- so when the offset is a known multiple of the
        // fragment width LLVM raises the alignment itself and the load becomes a single
        // wide access. When it cannot prove that -- a padded stride, say -- it splits the
        // vector back into element-sized ones, which is what this path would have emitted
        // by hand anyway.
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
        let align = scalar_alignment(ctx, ty.elem_ty);
        let lane = LanePosition::current(ctx, rw);

        let frag = load_fragment(ctx, rw, matrix, frag_ty);

        // The counterpart of the vector load in `LoadOp`; see the note there.
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
}

/// The `k` one instruction of this device's WMMA covers, for reporting a tile that does not
/// divide into whole ones.
fn instruction_k(ctx: &Context) -> usize {
    match ctx.wmma() {
        AmdWmma::Rdna3 => 16,
        AmdWmma::Rdna4 => 32,
    }
}

/// Names whichever of the two element types has no WMMA format.
fn unsupported_elem(ctx: &Context, ab: TypeHandle, cd: TypeHandle) -> MatrixElemUnsupported {
    let culprit = if wmma_format(ctx, ab).is_none() {
        ab
    } else {
        cd
    };
    MatrixElemUnsupported(culprit.disp(ctx).to_string())
}

/// The WMMA format name of a fragment element type, or `None` for one no instruction takes.
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

/// The `step`th `width`-element slice of a fragment.
///
/// Only reached when a tile takes more than one instruction, which is RDNA3 alone; there a
/// lane holds the whole `k` range of its row or column contiguously, so consecutive slices
/// of the fragment are exactly the operands of the consecutive instructions.
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

/// `fragment` re-indexed by `mask`, which may reorder, narrow or widen it.
fn shuffle(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    mask: Vec<i32>,
) -> Value {
    let op = llvm::ShuffleVectorOp::new(ctx, fragment, fragment, mask);
    insert(ctx, rw, &op)
}

/// What one WMMA lowering needs that is not the operand values.
struct WmmaCall {
    /// `m`, `n` and the depth of the *tile*, which may be several instructions.
    shape: MatrixShape,
    /// WMMA format name of the A/B element type.
    ab: &'static str,
    /// WMMA format name of the C/D element type.
    cd: &'static str,
    /// Whether the accumulator is a 16 bit type, which RDNA3 gives an `opsel` argument.
    cd_is_half: bool,
}

/// Emit the instructions a tile takes and return the final accumulator.
///
/// `ab_ty` types the A and B operands as the caller holds them, covering the whole tile; when
/// the tile is more than one instruction each step takes its own slice and is typed by that.
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

        // The intrinsic is overloaded on both fragment types, so the name carries them.
        let name = format!(
            "llvm.amdgcn.wmma.{cd}.{m}x{n}x{instruction_k}.{ab}.{}.{}",
            llvm_mangled_ty(ctx, cd_ty),
            llvm_mangled_ty(ctx, arg_ty),
        );

        let mut args = vec![a_arg, b_arg, acc];
        let mut arg_tys = vec![arg_ty, arg_ty, cd_ty];
        // RDNA3 writes a 16 bit result into one half of each register and takes an `opsel`
        // saying which. RDNA4 packs them densely and has no such argument.
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

        let (Some(ab), Some(cd)) = (
            wmma_format(ctx, a_ty.elem_ty),
            wmma_format(ctx, c_ty.elem_ty),
        ) else {
            return input_err!(
                self.loc(ctx),
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
            return input_err!(self.loc(ctx), MatrixDepthUnsupported(k, instruction_k(ctx)));
        };
        store_fragment(ctx, rw, d, result);

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

        let value = load_fragment(ctx, rw, input, in_frag_ty);

        // The two fragments hold the same *elements*, but not necessarily the same number of
        // vector slots: an RDNA3 accumulator pads a 16 bit element out to one per 32 bit
        // register, so it has two slots per element where a 32 bit one has a single slot.
        // Casting the raw vectors would hand `fptrunc` an <8 x float> and a <16 x half>, whose
        // shapes do not match and which it rejects. Gather to the dense elements, cast those,
        // and re-pad for the destination.
        let (elems, in_step) = fragment_layout(ctx, &in_ty);
        let (out_elems, out_step) = fragment_layout(ctx, &out_ty);

        // A and B lay their elements out the same way -- one row or column per lane, the `k`
        // range along it -- so a cast between those two idents only reinterprets which
        // operand the fragment is. An accumulator does not: it holds several rows per lane,
        // in a different count and a different order, so a cast across that boundary needs a
        // cross-lane relayout this lowering does not do. `cmma::cast_with_ident` can ask for
        // one, and the counts sometimes even agree (RDNA4, `k` of 16), so it is refused here
        // rather than left to write elements into the wrong positions.
        if (in_ty.ident == MatrixIdent::Accumulator) != (out_ty.ident == MatrixIdent::Accumulator) {
            return input_err!(
                self.loc(ctx),
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
            // Nothing to convert. The frontend folds this away, but the lowering does not
            // depend on it having done so.
            dense
        } else if is_half(ctx, in_ty.elem_ty) && is_half(ctx, out_ty.elem_ty) {
            // The one pair of the same width that LLVM holds in two different types: f16 and
            // bf16 split their 16 bits between exponent and mantissa differently, so neither
            // `fptrunc` nor `fpext` applies and keeping the bits would change the value they
            // stand for. f32 holds either exactly, so the conversion goes through it.
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
            // Any other pair of the same width is two cubecl names for one LLVM type -- f32,
            // flex32 and tf32 are all `float` -- so there is nothing to emit. Nothing could
            // be, either: `fpext` and `fptrunc` each want a change of width, and handing one
            // a source and a destination of the same type is invalid IR rather than a no-op.
            // Reaching here at all takes a fragment type the device advertises, which
            // `Matrix::uninitialized` checks before this lowering runs, so today only f16 and
            // bf16 share a width. The arm is what keeps a future third one from silently
            // taking the conversion above.
            debug_assert_eq!(
                cube_type_to_llvm(ctx, in_ty.elem_ty),
                cube_type_to_llvm(ctx, out_ty.elem_ty),
                "a cast of the same width between two distinct LLVM types needs a conversion, \
                 and neither `fpext` nor `fptrunc` is one"
            );
            dense
        };

        // Repeating each element across its slots rather than leaving the padding undefined:
        // only the low half of each register is ever read back, so the value there is free,
        // and a defined one keeps the fragment printable and comparable.
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
}

/// Widens every element of `value` to `ty`.
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

/// Narrows every element of `value` to `ty`.
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
    let lane = LanePosition::of(ctx, rw, lane);
    let along = along(ctx, rw, matrix, i, lane);

    // A is indexed by its row and walks `k`; B and the accumulator are the other way round.
    let lane_names_row = matrix.ident == MatrixIdent::A;
    match (axis, lane_names_row) {
        (Axis::Row, true) | (Axis::Col, false) => lane.in_row,
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

        let (Some(ab), Some(cd)) = (wmma_format(ctx, ab_elem), wmma_format(ctx, cd_elem)) else {
            return input_err!(self.loc(ctx), unsupported_elem(ctx, ab_elem, cd_elem));
        };
        let shape = *self.shape(ctx).clone();
        let call = WmmaCall {
            shape,
            ab,
            cd,
            cd_is_half: is_half(ctx, cd_elem),
        };
        let Some(result) = emit_wmma(ctx, rw, call, (a_val, b_val, c_val), ab_ty, cd_ty) else {
            return input_err!(
                self.loc(ctx),
                MatrixDepthUnsupported(shape.k, instruction_k(ctx))
            );
        };
        store_fragment(ctx, rw, d, result);

        rw.erase_operation(ctx, old_op);
        Ok(())
    }
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

    /// The A/B fragment width is what selects the WMMA intrinsic, and getting it wrong fails
    /// to select rather than computing a wrong answer. RDNA3 hands every lane the whole `k`
    /// range; RDNA4 splits it between the halves of the wave.
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

    /// RDNA3 writes a 16 bit accumulator into the low half of each register and leaves the
    /// other half alone, so its elements sit every second slot. A 32 bit one is dense, and so
    /// is every RDNA4 accumulator.
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

    /// gfx11 has only the 16-deep instruction, but the device properties advertise a 32-deep
    /// tile because rocWMMA implements one as two instructions. A deeper tile is therefore
    /// several instructions chained through the accumulator, not an error.
    #[test]
    fn a_tile_deeper_than_the_instruction_is_several_of_them() {
        assert_eq!(instruction_steps(AmdWmma::Rdna3, 16), Some((16, 1)));
        assert_eq!(instruction_steps(AmdWmma::Rdna3, 32), Some((16, 2)));
        assert_eq!(instruction_steps(AmdWmma::Rdna4, 32), Some((32, 1)));
    }

    /// A depth that is not a whole number of instructions has no lowering, and says so
    /// rather than emitting one that covers part of the tile.
    #[test]
    fn a_tile_that_does_not_divide_has_no_lowering() {
        assert_eq!(instruction_steps(AmdWmma::Rdna3, 24), None);
        assert_eq!(instruction_steps(AmdWmma::Rdna4, 0), None);
    }

    /// Whichever axis the layout makes contiguous is the one the stride does not multiply.
    /// A and B disagree about which that is, and the accumulator follows B.
    #[test]
    fn the_layout_decides_which_axis_carries_the_stride() {
        use MatrixIdent::{A, Accumulator, B};
        use MatrixLayout::{ColMajor, RowMajor};

        assert!(is_strided(A, ColMajor) && !is_strided(A, RowMajor));
        assert!(is_strided(B, RowMajor) && !is_strided(B, ColMajor));
        assert!(is_strided(Accumulator, RowMajor) && !is_strided(Accumulator, ColMajor));
    }

    /// A fragment loads as one wide access only when both its memory side and its register
    /// side step by one. An RDNA3 accumulator walks two rows at a time because the halves of
    /// the wave interleave, and a padded one leaves every second register slot alone, so
    /// neither can be filled by a contiguous load.
    #[test]
    fn only_a_fragment_dense_on_both_sides_loads_as_one_access() {
        let mut ctx = Context::default();
        let (f16, f32) = (f16(&mut ctx), f32(&mut ctx));
        let a = matrix(MatrixIdent::A, f16, 16);
        let acc16 = matrix(MatrixIdent::Accumulator, f16, 16);
        let acc32 = matrix(MatrixIdent::Accumulator, f32, 16);

        // A is contiguous whenever the layout leaves it unstrided.
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

        // An RDNA3 accumulator never is, padded or not; an RDNA4 one is when it is dense.
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
