use crate::{
    amdgpu::{
        fragment::{
            BF16, E4M3, E5M2, F16, F32, I8, I32, INSTRUCTION_M, INSTRUCTION_N, TILE_DEPTHS, U8,
            fp8_name, fragment_slice, instruction_steps, shuffle,
        },
        mfma::{self, CodeReading, MfmaForm, emit_mfma},
        plane::lane_id,
    },
    prelude::*,
    shared::matrix::{convert_lanes, elem_type, registers_as_vector, registers_value},
    shared::plane::bitcast,
};
use pliron::location::Location;

use cubecl_core::ir::{
    CoordinatePart, ElemType, ElementIndex,
    amd::{AmdMatrixCore, AmdMfma, AmdWmma},
    dialect::matrix::{
        CastOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RegistersOp,
        StMatrixOp, StoreOp,
    },
    features::MmaConfig,
    types::{MatrixIdent, MatrixLayout, MatrixShape, matrix::MatrixType},
};

#[derive(Debug, Error)]
#[error(
    "casting a {from} fragment to a {to} one needs a cross-lane relayout the AMDGPU lowering \
     does not implement: an accumulator spreads several rows over each lane, an A or B fragment \
     one"
)]
pub struct MatrixRelayoutUnsupported {
    pub from: MatrixIdent,
    pub to: MatrixIdent,
}

#[derive(Debug, Error)]
#[error(
    "a fragment's lane holds {held} cells, but the registers it is copied into hold {registers}"
)]
pub struct RegisterCountMismatch {
    pub held: usize,
    pub registers: usize,
}

#[derive(Debug, Error)]
#[error(
    "no matrix lowering for a {}x{}x{} tile: one is {INSTRUCTION_M}x{INSTRUCTION_N} and a \
     multiple of {instruction_k} deep",
    shape.m, shape.n, shape.k
)]
pub struct MatrixShapeUnsupported {
    pub shape: MatrixShape,
    pub instruction_k: usize,
}

#[derive(Debug, Error)]
#[error("no {core:?} instruction multiplies {a:?} by {b:?} into {cd:?}")]
pub struct MatrixElemUnsupported {
    pub core: AmdMatrixCore,
    pub a: ElemType,
    pub b: ElemType,
    pub cd: ElemType,
}

impl MatrixElemUnsupported {
    /// The refusal of a product of these operand types on the context's matrix core.
    fn of(ctx: &Context, a: TypeHandle, b: TypeHandle, cd: TypeHandle) -> Self {
        Self {
            core: matrix_core(ctx),
            a: elem_type(ctx, a),
            b: elem_type(ctx, b),
            cd: elem_type(ctx, cd),
        }
    }
}

/// RDNA3 pads 16-bit accumulators to 32-bit registers.
fn pads_half_accumulator(generation: AmdWmma) -> bool {
    generation == AmdWmma::Rdna3
}

/// The depth of every WMMA form this lowering emits, on RDNA3 and RDNA4 alike.
const INSTRUCTION_K: usize = 16;

/// The WMMA instruction a tile's element types select.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmmaForm {
    /// `f16` or `bf16` operands, accumulating in their own type or `f32`.
    Half { ab: ElemType, cd: ElemType },
    /// `e4m3` or `e5m2` operands, each side its own format, accumulating in `f32`. RDNA4 only.
    Fp8 { a: ElemType, b: ElemType },
    /// 8-bit integer operands, each side signed or not, accumulating in `i32`.
    Int8 { a_signed: bool, b_signed: bool },
}

impl WmmaForm {
    /// The form multiplying `a` by `b` into `cd` on `generation`, where it has one.
    pub fn of(generation: AmdWmma, a: ElemType, b: ElemType, cd: ElemType) -> Option<Self> {
        match (a, b, cd) {
            (F16 | BF16, _, _) if b == a && (cd == a || cd == F32) => {
                Some(WmmaForm::Half { ab: a, cd })
            }
            (E4M3 | E5M2, E4M3 | E5M2, F32) if generation == AmdWmma::Rdna4 => {
                Some(WmmaForm::Fp8 { a, b })
            }
            (I8 | U8, I8 | U8, I32) => Some(WmmaForm::Int8 {
                a_signed: a == I8,
                b_signed: b == I8,
            }),
            _ => None,
        }
    }

    /// Every tile `generation` lowers: each form at a 16x16 output, one or two instructions deep.
    pub fn configs(generation: AmdWmma) -> Vec<MmaConfig> {
        let halves =
            [(F16, F16), (F16, F32), (BF16, BF16), (BF16, F32)].map(|(ab, cd)| (ab, ab, cd));
        let fp8 = [E4M3, E5M2].map(|a| [E4M3, E5M2].map(|b| (a, b, F32)));
        let int8 = [I8, U8].map(|a| [I8, U8].map(|b| (a, b, I32)));
        let elems = halves
            .into_iter()
            .chain(fp8.into_iter().flatten())
            .chain(int8.into_iter().flatten());

        elems
            .filter(|&(a, b, cd)| Self::of(generation, a, b, cd).is_some())
            .flat_map(|(a, b, cd)| {
                TILE_DEPTHS.map(|depth| MmaConfig {
                    a_type: a,
                    b_type: b,
                    cd_type: cd,
                    m: INSTRUCTION_M as u32,
                    n: INSTRUCTION_N as u32,
                    k: (depth * INSTRUCTION_K) as u32,
                })
            })
            .collect()
    }
}

impl WmmaForm {
    /// The form multiplying B by A where this one multiplies A by B.
    fn swapped(self) -> Self {
        match self {
            WmmaForm::Half { .. } => self,
            WmmaForm::Fp8 { a, b } => WmmaForm::Fp8 { a: b, b: a },
            WmmaForm::Int8 { a_signed, b_signed } => WmmaForm::Int8 {
                a_signed: b_signed,
                b_signed: a_signed,
            },
        }
    }

    /// The intrinsic's name up to its overloaded types.
    fn intrinsic(&self) -> String {
        let shape = format!("{INSTRUCTION_M}x{INSTRUCTION_N}x{INSTRUCTION_K}");
        match *self {
            WmmaForm::Half { ab, cd } => {
                format!(
                    "llvm.amdgcn.wmma.{}.{shape}.{}",
                    half_name(cd),
                    half_name(ab)
                )
            }
            WmmaForm::Fp8 { a, b } => {
                format!(
                    "llvm.amdgcn.wmma.f32.{shape}.{}.{}",
                    fp8_name(a),
                    fp8_name(b)
                )
            }
            WmmaForm::Int8 { .. } => format!("llvm.amdgcn.wmma.i32.{shape}.iu8"),
        }
    }

    /// The `i1` operands the intrinsic takes around its fragments.
    ///
    /// Every 16-bit accumulator form takes `op_sel` last: RDNA3 selects a register half with it,
    /// RDNA4 packs its accumulators and requires it to be 0. The integer forms take each
    /// operand's signedness ahead of it, and a clamp last, off: an `i32` sum wraps.
    fn flags(&self) -> WmmaFlags {
        match *self {
            WmmaForm::Half { cd, .. } if cd != F32 => WmmaFlags {
                last: Some(false),
                ..WmmaFlags::default()
            },
            WmmaForm::Half { .. } | WmmaForm::Fp8 { .. } => WmmaFlags::default(),
            WmmaForm::Int8 { a_signed, b_signed } => WmmaFlags {
                before_a: Some(a_signed),
                before_b: Some(b_signed),
                last: Some(false),
            },
        }
    }
}

fn half_name(elem: ElemType) -> &'static str {
    match elem {
        F16 => "f16",
        BF16 => "bf16",
        F32 => "f32",
        other => unreachable!("{other:?} is no half precision WMMA element"),
    }
}

/// The `i1` operands a WMMA intrinsic takes ahead of A, ahead of B and after the accumulator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WmmaFlags {
    pub before_a: Option<bool>,
    pub before_b: Option<bool>,
    pub last: Option<bool>,
}

/// The matrix core the kernel being lowered runs on.
///
/// # Panics
///
/// - Where the device has neither WMMA nor MFMA: a matrix op is lowered only for one that has.
pub fn matrix_core(ctx: &Context) -> AmdMatrixCore {
    *ctx.aux_ty::<Option<AmdMatrixCore>>()
        .as_ref()
        .expect("a matrix op should only be compiled for a device that has WMMA or MFMA")
}

/// Sets the matrix core the kernel being lowered runs on, `None` for a device with neither.
pub fn set_matrix_core(ctx: &mut Context, core: Option<AmdMatrixCore>) {
    ctx.set_aux_ty(core);
}

/// Accumulator elements each lane holds on WMMA: a 16x16 output over 32 lanes.
const ACCUMULATOR_REGISTER_COUNT: usize = 8;

/// Fragment size and element spacing per lane.
fn fragment_layout(ctx: &Context, matrix: &MatrixType) -> (usize, usize) {
    let core = matrix_core(ctx);
    match (matrix.ident, core) {
        (MatrixIdent::A | MatrixIdent::B, _) => (core.frag_ab_elems_at_depth(matrix.shape.k), 1),
        (MatrixIdent::Accumulator, AmdMatrixCore::Wmma(generation)) => {
            accumulator_layout(ctx, generation, matrix.elem_ty)
        }
        (MatrixIdent::Accumulator, AmdMatrixCore::Mfma(_)) => (mfma::ACCUMULATOR_REGISTERS, 1),
    }
}

/// A WMMA accumulator's size and element spacing per lane: RDNA3 spreads a 16-bit one over every
/// other slot.
fn accumulator_layout(ctx: &Context, generation: AmdWmma, elem: TypeHandle) -> (usize, usize) {
    let half = matches!(elem_type(ctx, elem), F16 | BF16);
    let padded = half && pads_half_accumulator(generation);
    (ACCUMULATOR_REGISTER_COUNT, if padded { 2 } else { 1 })
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
    pointee.expect("a matrix operand should point at a matrix")
}

fn mul(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

pub fn add(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

/// The coordinate `parts` give element `nth` of the current lane, `lane`.
fn coordinate(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    parts: &[CoordinatePart],
    lane: Value,
    nth: usize,
) -> Value {
    // Lane 0 leaves the parts its lane does not count: the element's.
    let of_element: usize = parts.iter().map(|part| part.of_nth(0, nth)).sum();
    let of_element = insert_i32_const(ctx, rw, of_element as i32);
    let of_lane = parts.iter().filter(|part| part.index == ElementIndex::Lane);
    let mut sum = of_element;
    for part in of_lane {
        let run = insert_i32_const(ctx, rw, part.run as i32);
        let step = llvm::UDivOp::new(ctx, lane, run);
        let step = insert(ctx, rw, &step);
        let step = match part.count {
            Some(count) => counted_over(ctx, rw, step, count),
            None => step,
        };
        let stride = insert_i32_const(ctx, rw, part.stride as i32);
        let place = mul(ctx, rw, step, stride);
        sum = add(ctx, rw, sum, place);
    }
    sum
}

fn counted_over(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    step: Value,
    count: usize,
) -> Value {
    let count = insert_i32_const(ctx, rw, count as i32);
    let op = llvm::URemOp::new(ctx, step, count);
    insert(ctx, rw, &op)
}

/// Where element `nth` of the fragment of the current lane, `lane`, lies in memory laid out as
/// `layout`.
fn element_index(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: &MatrixType,
    layout: MatrixLayout,
    nth: usize,
    lane: Value,
    stride: Value,
) -> Value {
    let core = matrix_core(ctx);
    let positions = core.element_positions(matrix.ident, matrix.shape);
    let row = coordinate(ctx, rw, &positions.row, lane, nth);
    let col = coordinate(ctx, rw, &positions.col, lane, nth);
    let (along, across) = match core.lane_names_row(matrix.ident) {
        true => (col, row),
        false => (row, col),
    };

    let (scaled, unscaled) = if is_strided(core, matrix.ident, layout) {
        (along, across)
    } else {
        (across, along)
    };
    let scaled = mul(ctx, rw, scaled, stride);
    add(ctx, rw, scaled, unscaled)
}

/// Whether a lane's run of elements steps by the stride in memory laid out as `layout`.
fn is_strided(core: AmdMatrixCore, ident: MatrixIdent, layout: MatrixLayout) -> bool {
    layout != core.run_layout(ident)
}

/// Vector access requires consecutive elements in both memory and registers.
fn fragment_is_contiguous(
    core: AmdMatrixCore,
    matrix: &MatrixType,
    layout: MatrixLayout,
    step: usize,
) -> bool {
    if step != 1 || is_strided(core, matrix.ident, layout) {
        return false;
    }
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => true,
        MatrixIdent::Accumulator => core != AmdMatrixCore::Wmma(AmdWmma::Rdna3),
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

pub fn store_fragment(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    value: Value,
) {
    let op = llvm::StoreOp::new(ctx, value, matrix);
    rw.insert_op(ctx, &op);
}

/// What a load or a store of a fragment of `matrix` reads off it: its matrix type, its element
/// count and spacing, its fragment and element types, the alignment a tile access has, and the
/// current lane.
fn tile_access(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
    matrix: Value,
) -> (MatrixType, usize, usize, TypeHandle, TypeHandle, u32, Value) {
    let ty = matrix_of(ctx, operands_info, matrix);
    let (elems, step) = fragment_layout(ctx, &ty);
    let frag_ty = fragment_ty(ctx, &ty);
    let elem_ty = cube_type_to_llvm(ctx, ty.elem_ty);
    let align = scalar_alignment(ctx, ty.elem_ty);
    let lane = lane_id(ctx, rw);
    (ty, elems, step, frag_ty, elem_ty, align, lane)
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
    let (ty, elems, step, frag_ty, elem_ty, align, lane) =
        tile_access(ctx, rw, operands_info, matrix);

    if fragment_is_contiguous(matrix_core(ctx), &ty, layout, step) {
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
    let (ty, elems, step, frag_ty, elem_ty, align, lane) =
        tile_access(ctx, rw, operands_info, matrix);

    let frag = load_fragment(ctx, rw, matrix, frag_ty);

    if fragment_is_contiguous(matrix_core(ctx), &ty, layout, step) {
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

/// The instruction a tile's element types select on the part's matrix core.
#[derive(Clone, Copy, Debug)]
pub enum MatrixForm {
    Wmma(WmmaForm),
    Mfma(MfmaForm, AmdMfma),
}

impl MatrixForm {
    /// The form a plain MMA of these element types lowers to on the context's matrix core, or
    /// why there is none. A form reading its 8-bit floats as FNUZ is none: it would read OCP
    /// bytes at half their value and a negative zero as NaN.
    pub fn plain(
        ctx: &Context,
        a: TypeHandle,
        b: TypeHandle,
        cd: TypeHandle,
    ) -> core::result::Result<Self, MatrixElemUnsupported> {
        let form = Self::any(ctx, a, b, cd)?;
        match form.code_reading() {
            CodeReading::Ocp => Ok(form),
            CodeReading::Fnuz => Err(MatrixElemUnsupported::of(ctx, a, b, cd)),
        }
    }

    /// How the form reads its 8-bit float bytes.
    pub fn code_reading(&self) -> CodeReading {
        match self {
            MatrixForm::Wmma(_) => CodeReading::Ocp,
            MatrixForm::Mfma(form, _) => form.code_reading(),
        }
    }

    /// The form multiplying these element types on the context's matrix core, FNUZ forms among
    /// them, or why there is none.
    pub fn any(
        ctx: &Context,
        a: TypeHandle,
        b: TypeHandle,
        cd: TypeHandle,
    ) -> core::result::Result<Self, MatrixElemUnsupported> {
        let (a_elem, b_elem, cd_elem) = (elem_type(ctx, a), elem_type(ctx, b), elem_type(ctx, cd));
        Self::of(matrix_core(ctx), a_elem, b_elem, cd_elem)
            .ok_or_else(|| MatrixElemUnsupported::of(ctx, a, b, cd))
    }

    /// The form multiplying `a` by `b` into `cd` on `core`, FNUZ forms among them, where it has
    /// one.
    pub fn of(core: AmdMatrixCore, a: ElemType, b: ElemType, cd: ElemType) -> Option<Self> {
        match core {
            AmdMatrixCore::Wmma(generation) => {
                WmmaForm::of(generation, a, b, cd).map(MatrixForm::Wmma)
            }
            AmdMatrixCore::Mfma(generation) => {
                MfmaForm::of(generation, a, b, cd).map(|form| MatrixForm::Mfma(form, generation))
            }
        }
    }

    /// How deep one instruction is.
    pub fn instruction_k(&self) -> usize {
        match self {
            MatrixForm::Wmma(_) => INSTRUCTION_K,
            MatrixForm::Mfma(form, generation) => form.instruction_k(*generation),
        }
    }
}

/// Emits the instructions of `form` for a tile, or `None` where no whole number of them makes
/// its shape.
pub fn emit(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    form: MatrixForm,
    shape: MatrixShape,
    operands: (Value, Value, Value),
    cd_ty: TypeHandle,
) -> Option<Value> {
    match form {
        MatrixForm::Wmma(form) => emit_wmma(ctx, rw, form, shape, operands, cd_ty),
        MatrixForm::Mfma(form, generation) => {
            emit_mfma(ctx, rw, (form, generation), shape, operands)
        }
    }
}

/// The plain form multiplying these element types on the context's matrix core, or its refusal
/// at `loc`.
fn plain_form(
    ctx: &Context,
    loc: Location,
    a: TypeHandle,
    b: TypeHandle,
    cd: TypeHandle,
) -> Result<MatrixForm> {
    MatrixForm::plain(ctx, a, b, cd).map_err(|err| pliron::input_error!(loc, err))
}

/// The product of `operands` by `form`, or the refusal at `loc` where no whole number of its
/// instructions makes `shape`.
#[allow(clippy::too_many_arguments)]
fn emit_product(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    loc: Location,
    form: MatrixForm,
    shape: MatrixShape,
    operands: (Value, Value, Value),
    cd_ty: TypeHandle,
) -> Result<Value> {
    let unsupported = MatrixShapeUnsupported {
        shape,
        instruction_k: form.instruction_k(),
    };
    emit(ctx, rw, form, shape, operands, cd_ty).map_or_else(|| input_err!(loc, unsupported), Ok)
}

/// Emits the instructions for a tile, or `None` where no whole number of them makes its shape.
///
/// A deeper tile is cut into consecutive slices of each lane's fragment. On RDNA4 a lane holds
/// its half's run of `k`, so a slice pairs `k` from both halves the instruction would not, but A
/// and B are sliced alike and the products summed are the same.
///
/// On RDNA4, A and B hold their registers alike, so the instruction is given B as its A and A as
/// its B: it computes the transposed product, which leaves each lane one row of the accumulator,
/// as A holds it (see `AmdMatrixCore::mma_properties`).
fn emit_wmma(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    form: WmmaForm,
    shape: MatrixShape,
    (a_val, b_val, c_val): (Value, Value, Value),
    cd_ty: TypeHandle,
) -> Option<Value> {
    let steps = instruction_steps(shape, INSTRUCTION_K)?;
    let (form, a_val, b_val) = match matrix_core(ctx) {
        AmdMatrixCore::Wmma(AmdWmma::Rdna4) => (form.swapped(), b_val, a_val),
        AmdMatrixCore::Wmma(AmdWmma::Rdna3) | AmdMatrixCore::Mfma(_) => (form, a_val, b_val),
    };
    let slice_width = matrix_core(ctx).frag_ab_elems_at_depth(INSTRUCTION_K);

    let fragment_cd_ty = cd_ty;
    let c_val = wmma_operand(ctx, rw, c_val);
    let cd_ty = c_val.get_type(ctx);

    // The flags are the same for every instruction, so each is built once.
    let flags = form.flags();
    let mut flag = |value: Option<bool>| value.map(|value| insert_bool_const(ctx, rw, value));
    let (before_a, before_b, last) = (flag(flags.before_a), flag(flags.before_b), flag(flags.last));

    let mut acc = c_val;
    for step in 0..steps {
        let (a_arg, b_arg) = if steps == 1 {
            (a_val, b_val)
        } else {
            fragment_slices(ctx, rw, (a_val, b_val), step, slice_width)
        };
        let a_arg = wmma_operand(ctx, rw, a_arg);
        let b_arg = wmma_operand(ctx, rw, b_arg);
        let arg_ty = a_arg.get_type(ctx);

        let name = format!(
            "{}.{}.{}",
            form.intrinsic(),
            llvm_mangled_ty(ctx, cd_ty),
            llvm_mangled_ty(ctx, arg_ty),
        );

        let args: Vec<Value> = before_a
            .into_iter()
            .chain([a_arg])
            .chain(before_b)
            .chain([b_arg, acc])
            .chain(last)
            .collect();
        let arg_tys = args.iter().map(|arg| arg.get_type(ctx)).collect();

        let fn_ty = FuncType::get(ctx, cd_ty, arg_tys, false);
        let op = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
        acc = insert(ctx, rw, &op);
    }
    Some(bitcast(ctx, rw, acc, fragment_cd_ty))
}

/// Slice `step` of A's and of B's fragment, each `width` elements.
fn fragment_slices(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    (a, b): (Value, Value),
    step: usize,
    width: usize,
) -> (Value, Value) {
    let a = fragment_slice(ctx, rw, a, step, width);
    (a, fragment_slice(ctx, rw, b, step, width))
}

/// A fragment as the WMMA intrinsics take it: the `bf16` forms declare their lanes as `i16`, and
/// the 8-bit forms pack four lanes into each `i32`.
fn wmma_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, fragment: Value) -> Value {
    let ty = fragment.get_type(ctx);
    let (elem, lanes) = {
        let vector = ty.deref(ctx);
        let vector = vector
            .downcast_ref::<LlvmVectorType>()
            .expect("a fragment should be held in a vector");
        (vector.elem_type(), vector.num_elements())
    };
    let Some((width, lanes)) = wmma_packing(ctx, elem, lanes) else {
        return fragment;
    };
    let packed = IntegerType::get(ctx, width, Signedness::Signless).into();
    let packed_ty = LlvmVectorType::get(ctx, packed, lanes, VectorTypeKind::Fixed).into();
    bitcast(ctx, rw, fragment, packed_ty)
}

/// The integer width and lane count a WMMA intrinsic declares a fragment of `lanes` elements of
/// `elem` in, `None` where it takes the fragment as it is.
fn wmma_packing(ctx: &Context, elem: TypeHandle, lanes: u32) -> Option<(u32, u32)> {
    let elem = elem.deref(ctx);
    let bf16 = elem.is::<BF16Type>();
    let byte = elem
        .downcast_ref::<IntegerType>()
        .is_some_and(|int| int.width() == 8);
    match (bf16, byte) {
        (true, _) => Some((16, lanes)),
        (false, true) => Some((32, lanes / 4)),
        (false, false) => None,
    }
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
    let b_ty = matrix_of(ctx, operands_info, b);
    let c_ty = matrix_of(ctx, operands_info, c);
    let a_frag_ty = fragment_ty(ctx, &a_ty);
    let b_frag_ty = fragment_ty(ctx, &b_ty);
    let cd_frag_ty = fragment_ty(ctx, &c_ty);

    let loc = op.loc(ctx);
    let form = plain_form(ctx, loc.clone(), a_ty.elem_ty, b_ty.elem_ty, c_ty.elem_ty)?;

    let a_val = load_fragment(ctx, rw, a, a_frag_ty);
    let b_val = load_fragment(ctx, rw, b, b_frag_ty);
    let c_val = load_fragment(ctx, rw, c, cd_frag_ty);

    let operands = (a_val, b_val, c_val);
    let result = emit_product(ctx, rw, loc, form, a_ty.shape, operands, cd_frag_ty)?;
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
        let unsupported = MatrixRelayoutUnsupported {
            from: in_ty.ident,
            to: out_ty.ident,
        };
        return input_err!(op.loc(ctx), unsupported);
    }
    assert_eq!(
        elems, out_elems,
        "a cast keeps the element count and changes only their width"
    );

    let gathered = (0..elems).map(|i| (i * in_step) as i32).collect();
    let dense = regather(ctx, rw, value, in_step, gathered);

    let cast = convert_lanes(ctx, rw, dense, out_ty.elem_ty);

    let spread = (0..elems * out_step)
        .map(|i| (i / out_step) as i32)
        .collect();
    let result = regather(ctx, rw, cast, out_step, spread);
    store_fragment(ctx, rw, output, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// The cells a lane holds of a fragment copied into its registers, element `i` of the lane's
/// [`element_positions`](AmdMatrixCore::element_positions) at register cell `i`: the fragment's
/// slot `i * step`, past a padded accumulator's unused halves.
pub(crate) fn registers(
    op: &RegistersOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let matrix = op.matrix(ctx);
    let registers = op.registers(ctx);

    let ty = matrix_of(ctx, operands_info, matrix);
    let (elems, step) = fragment_layout(ctx, &ty);
    let (registers_ty, _) = registers_as_vector(ctx, operands_info, registers);
    let lanes = registers_ty
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.num_elements() as usize);
    if lanes != Some(elems) {
        let mismatch = RegisterCountMismatch {
            held: elems,
            registers: lanes.unwrap_or(0),
        };
        return input_err!(op.loc(ctx), mismatch);
    }

    let frag_ty = fragment_ty(ctx, &ty);
    let value = load_fragment(ctx, rw, matrix, frag_ty);
    let slots = (0..elems).map(|i| (i * step) as i32).collect();
    let dense = regather(ctx, rw, value, step, slots);
    store_fragment(ctx, rw, registers, dense);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// `value` with its lanes taken from `slots`, or as it is where its elements are `step` 1 apart.
fn regather(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    step: usize,
    slots: Vec<i32>,
) -> Value {
    if step == 1 {
        value
    } else {
        shuffle(ctx, rw, value, slots)
    }
}

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

    let (a_ty, a_elem) = registers_as_vector(ctx, operands_info, a);
    let (b_ty, b_elem) = registers_as_vector(ctx, operands_info, b);
    let (cd_ty, cd_elem) = registers_as_vector(ctx, operands_info, c);

    let loc = op.loc(ctx);
    let form = plain_form(ctx, loc.clone(), a_elem, b_elem, cd_elem)?;

    let a_val = registers_value(ctx, rw, a, a_ty);
    let b_val = registers_value(ctx, rw, b, b_ty);
    let c_val = registers_value(ctx, rw, c, cd_ty);

    let shape = op.shape(ctx).0;
    let result = emit_product(ctx, rw, loc, form, shape, (a_val, b_val, c_val), cd_ty)?;
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
            set_matrix_core(&mut ctx, Some(AmdMatrixCore::Wmma(generation)));
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
            set_matrix_core(&mut ctx, Some(AmdMatrixCore::Wmma(generation)));
            let acc = matrix(MatrixIdent::Accumulator, elem, 16);
            assert_eq!(
                fragment_layout(&ctx, &acc),
                (ACCUMULATOR_REGISTER_COUNT, step),
                "{generation:?}"
            );
        }
    }

    #[test]
    fn fp8_is_an_rdna4_form() {
        assert_eq!(
            WmmaForm::of(AmdWmma::Rdna4, E5M2, E4M3, F32),
            Some(WmmaForm::Fp8 { a: E5M2, b: E4M3 })
        );
        assert_eq!(WmmaForm::of(AmdWmma::Rdna3, E4M3, E4M3, F32), None);
        assert_eq!(WmmaForm::of(AmdWmma::Rdna4, E4M3, E4M3, F16), None);
    }

    #[test]
    fn each_integer_operand_keeps_its_own_sign() {
        for generation in [AmdWmma::Rdna3, AmdWmma::Rdna4] {
            assert_eq!(
                WmmaForm::of(generation, U8, I8, I32),
                Some(WmmaForm::Int8 {
                    a_signed: false,
                    b_signed: true
                })
            );
        }
        assert_eq!(WmmaForm::of(AmdWmma::Rdna4, I8, I8, F32), None);
    }

    #[test]
    fn a_half_form_takes_matching_operands() {
        assert!(WmmaForm::of(AmdWmma::Rdna4, F16, F16, F16).is_some());
        assert!(WmmaForm::of(AmdWmma::Rdna4, BF16, BF16, F32).is_some());
        assert_eq!(WmmaForm::of(AmdWmma::Rdna4, F16, BF16, F32), None);
        assert_eq!(WmmaForm::of(AmdWmma::Rdna4, BF16, BF16, F16), None);
    }

    #[test]
    fn the_intrinsic_names_each_operand_format() {
        let fp8 = WmmaForm::Fp8 { a: E4M3, b: E5M2 };
        assert_eq!(fp8.intrinsic(), "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8");
        let half = WmmaForm::Half { ab: BF16, cd: BF16 };
        assert_eq!(half.intrinsic(), "llvm.amdgcn.wmma.bf16.16x16x16.bf16");
        assert_eq!(
            half.flags().last,
            Some(false),
            "a 16-bit accumulator takes op_sel"
        );
        let int8 = WmmaForm::Int8 {
            a_signed: true,
            b_signed: false,
        };
        assert_eq!(
            int8.flags(),
            WmmaFlags {
                before_a: Some(true),
                before_b: Some(false),
                last: Some(false),
            }
        );
    }

    #[test]
    fn rdna4_offers_fp8_and_both_generations_offer_int8() {
        let rdna3 = WmmaForm::configs(AmdWmma::Rdna3);
        let rdna4 = WmmaForm::configs(AmdWmma::Rdna4);
        // Four half forms and four integer sign pairs, plus four fp8 pairs on RDNA4, each at
        // two depths.
        assert_eq!(rdna3.len(), 8 * 2);
        assert_eq!(rdna4.len(), 12 * 2);
        assert!(rdna3.iter().all(|config| config.a_type != E4M3));
        assert!(rdna4.iter().all(|config| config.m == 16 && config.n == 16));
    }

    #[test]
    fn a_swapped_form_swaps_its_operands_formats_and_signs() {
        let fp8 = WmmaForm::Fp8 { a: E4M3, b: E5M2 };
        let int8 = WmmaForm::Int8 {
            a_signed: true,
            b_signed: false,
        };

        assert_eq!(fp8.swapped(), WmmaForm::Fp8 { a: E5M2, b: E4M3 });
        assert_eq!(
            int8.swapped(),
            WmmaForm::Int8 {
                a_signed: false,
                b_signed: true
            }
        );
    }

    #[test]
    fn only_a_fragment_dense_on_both_sides_loads_as_one_access() {
        let mut ctx = Context::default();
        let (f16, f32) = (f16(&mut ctx), f32(&mut ctx));
        let a = matrix(MatrixIdent::A, f16, 16);
        let acc16 = matrix(MatrixIdent::Accumulator, f16, 16);
        let acc32 = matrix(MatrixIdent::Accumulator, f32, 16);

        assert!(fragment_is_contiguous(
            AmdMatrixCore::Wmma(AmdWmma::Rdna3),
            &a,
            MatrixLayout::RowMajor,
            1
        ));
        assert!(!fragment_is_contiguous(
            AmdMatrixCore::Wmma(AmdWmma::Rdna3),
            &a,
            MatrixLayout::ColMajor,
            1
        ));

        assert!(!fragment_is_contiguous(
            AmdMatrixCore::Wmma(AmdWmma::Rdna3),
            &acc32,
            MatrixLayout::ColMajor,
            1
        ));
        assert!(fragment_is_contiguous(
            AmdMatrixCore::Wmma(AmdWmma::Rdna4),
            &acc32,
            MatrixLayout::RowMajor,
            1
        ));
        assert!(!fragment_is_contiguous(
            AmdMatrixCore::Wmma(AmdWmma::Rdna4),
            &acc16,
            MatrixLayout::RowMajor,
            2
        ));
    }
}

#[derive(Debug, Error)]
#[error(
    "`{op}` is `ldmatrix`/`stmatrix`, an NVIDIA instruction; the AMDGPU backend does not \
     advertise it and has nothing to lower it to"
)]
pub struct MatrixTileMoveUnsupported {
    pub op: &'static str,
}

macro_rules! unsupported_tile_move {
    ($fn_name:ident, $cube_op:ty) => {
        pub(crate) fn $fn_name(
            op: &$cube_op,
            ctx: &mut Context,
            _rw: &mut DialectConversionRewriter,
            _operands_info: &OperandsInfo,
        ) -> Result<()> {
            input_err!(
                op.loc(ctx),
                MatrixTileMoveUnsupported {
                    op: stringify!($cube_op)
                }
            )
        }
    };
}

unsupported_tile_move!(ld_matrix, LdMatrixOp);
unsupported_tile_move!(st_matrix, StMatrixOp);
