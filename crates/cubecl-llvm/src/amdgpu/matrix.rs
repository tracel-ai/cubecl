//! AMDGPU matrix operations.

use crate::{
    amdgpu::plane::lane_id,
    prelude::*,
    shared::matrix::{convert_lanes, elem_type, registers_as_vector, registers_value},
    shared::plane::bitcast,
};
use cubecl_core::ir::{
    ElemType, FloatKind, IntKind, UIntKind,
    amd::AmdWmma,
    dialect::matrix::{
        CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp,
        RowIndexOp, StMatrixOp, StoreOp,
    },
    features::MmaConfig,
    types::{MatrixIdent, MatrixLayout, MatrixShape, matrix::MatrixType},
};

#[derive(Debug, Error)]
#[error(
    "casting a {0} fragment to a {1} one needs a cross-lane relayout the AMDGPU lowering does \
     not implement: an accumulator spreads several rows over each lane, an A or B fragment one"
)]
pub struct MatrixRelayoutUnsupported(MatrixIdent, MatrixIdent);

#[derive(Debug, Error)]
#[error(
    "no WMMA lowering for a {}x{}x{} tile: one is {INSTRUCTION_M}x{INSTRUCTION_N} and a \
     multiple of {INSTRUCTION_K} deep",
    .0.m, .0.n, .0.k
)]
pub struct MatrixShapeUnsupported(MatrixShape);

#[derive(Debug, Error)]
#[error("no {0:?} WMMA instruction multiplies {1:?} by {2:?} into {3:?}")]
pub struct MatrixElemUnsupported(AmdWmma, ElemType, ElemType, ElemType);

/// RDNA3 pads 16-bit accumulators to 32-bit registers.
fn pads_half_accumulator(generation: AmdWmma) -> bool {
    generation == AmdWmma::Rdna3
}

/// The shape of every WMMA form this lowering emits, on RDNA3 and RDNA4 alike.
const INSTRUCTION_M: usize = 16;
const INSTRUCTION_N: usize = 16;
const INSTRUCTION_K: usize = 16;

/// The tile depths offered: one instruction, or two in a row.
const TILE_KS: [usize; 2] = [INSTRUCTION_K, 2 * INSTRUCTION_K];

/// How many instructions a tile takes, where its output is one instruction's and its depth a
/// whole number of them.
///
/// A deeper tile is cut into consecutive slices of each lane's fragment. On RDNA4 a lane holds
/// its half's run of `k`, so a slice pairs `k` from both halves the instruction would not, but A
/// and B are sliced alike and the products summed are the same.
fn instruction_steps(shape: MatrixShape) -> Option<usize> {
    let MatrixShape { m, n, k } = shape;
    let output = m == INSTRUCTION_M && n == INSTRUCTION_N;
    let depth = k > 0 && k.is_multiple_of(INSTRUCTION_K);
    (output && depth).then_some(k / INSTRUCTION_K)
}

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

const F16: ElemType = ElemType::Float(FloatKind::F16);
const BF16: ElemType = ElemType::Float(FloatKind::BF16);
const F32: ElemType = ElemType::Float(FloatKind::F32);
const E4M3: ElemType = ElemType::Float(FloatKind::E4M3);
const E5M2: ElemType = ElemType::Float(FloatKind::E5M2);
const I8: ElemType = ElemType::Int(IntKind::I8);
const U8: ElemType = ElemType::UInt(UIntKind::U8);
const I32: ElemType = ElemType::Int(IntKind::I32);

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
                TILE_KS.map(|k| MmaConfig {
                    a_type: a,
                    b_type: b,
                    cd_type: cd,
                    m: INSTRUCTION_M as u32,
                    n: INSTRUCTION_N as u32,
                    k: k as u32,
                })
            })
            .collect()
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

/// AMD names `e4m3` `fp8` and `e5m2` `bf8`.
fn fp8_name(elem: ElemType) -> &'static str {
    match elem {
        E4M3 => "fp8",
        E5M2 => "bf8",
        other => unreachable!("{other:?} is no fp8 WMMA element"),
    }
}

/// The `i1` operands a WMMA intrinsic takes ahead of A, ahead of B and after the accumulator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct WmmaFlags {
    before_a: Option<bool>,
    before_b: Option<bool>,
    last: Option<bool>,
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

const ACCUMULATOR_REGISTERS: usize = 8;

const LANES_PER_ROW: u32 = 16;

/// Fragment size and element spacing per lane.
fn fragment_layout(ctx: &Context, matrix: &MatrixType) -> (usize, usize) {
    let generation = ctx.wmma();
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B => (generation.frag_ab_elems(matrix.shape.k), 1),
        MatrixIdent::Accumulator => {
            let half = matches!(elem_type(ctx, matrix.elem_ty), F16 | BF16);
            let padded = half && pads_half_accumulator(generation);
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

/// The form multiplying these element types on the context's generation, or why there is none.
fn wmma_form(
    ctx: &Context,
    a: TypeHandle,
    b: TypeHandle,
    cd: TypeHandle,
) -> core::result::Result<WmmaForm, MatrixElemUnsupported> {
    let generation = ctx.wmma();
    let (a, b, cd) = (elem_type(ctx, a), elem_type(ctx, b), elem_type(ctx, cd));
    WmmaForm::of(generation, a, b, cd).ok_or(MatrixElemUnsupported(generation, a, b, cd))
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

/// Emits the instructions for a tile, or `None` where no whole number of them makes its shape.
fn emit_wmma(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    form: WmmaForm,
    shape: MatrixShape,
    (a_val, b_val, c_val): (Value, Value, Value),
    cd_ty: TypeHandle,
) -> Option<Value> {
    let steps = instruction_steps(shape)?;
    let slice_width = ctx.wmma().frag_ab_elems(INSTRUCTION_K);

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
            let a_slice = fragment_slice(ctx, rw, a_val, step, slice_width);
            let b_slice = fragment_slice(ctx, rw, b_val, step, slice_width);
            (a_slice, b_slice)
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

/// A fragment as the WMMA intrinsics take it: the `bf16` forms declare their lanes as `i16`, and
/// the 8-bit forms pack four lanes into each `i32`.
fn wmma_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, fragment: Value) -> Value {
    let ty = fragment.get_type(ctx);
    let (elem, lanes) = {
        let vector = ty.deref(ctx);
        let vector = vector
            .downcast_ref::<LlvmVectorType>()
            .expect("a fragment is held in a vector");
        (vector.elem_type(), vector.num_elements())
    };
    let (width, lanes) = if elem.deref(ctx).is::<BF16Type>() {
        (16, lanes)
    } else if elem
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|int| int.width() == 8)
    {
        (32, lanes / 4)
    } else {
        return fragment;
    };
    let packed = IntegerType::get(ctx, width, Signedness::Signless).into();
    let packed_ty = LlvmVectorType::get(ctx, packed, lanes, VectorTypeKind::Fixed).into();
    bitcast(ctx, rw, fragment, packed_ty)
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

    let form = match wmma_form(ctx, a_ty.elem_ty, b_ty.elem_ty, c_ty.elem_ty) {
        Ok(form) => form,
        Err(err) => return input_err!(op.loc(ctx), err),
    };

    let a_val = load_fragment(ctx, rw, a, a_frag_ty);
    let b_val = load_fragment(ctx, rw, b, b_frag_ty);
    let c_val = load_fragment(ctx, rw, c, cd_frag_ty);

    let shape = a_ty.shape;
    let Some(result) = emit_wmma(ctx, rw, form, shape, (a_val, b_val, c_val), cd_frag_ty) else {
        return input_err!(op.loc(ctx), MatrixShapeUnsupported(shape));
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

    let cast = convert_lanes(ctx, rw, dense, out_ty.elem_ty);

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

    let (a_ty, a_elem) = registers_as_vector(ctx, operands_info, a);
    let (b_ty, b_elem) = registers_as_vector(ctx, operands_info, b);
    let (cd_ty, cd_elem) = registers_as_vector(ctx, operands_info, c);

    let form = match wmma_form(ctx, a_elem, b_elem, cd_elem) {
        Ok(form) => form,
        Err(err) => return input_err!(op.loc(ctx), err),
    };

    let a_val = registers_value(ctx, rw, a, a_ty);
    let b_val = registers_value(ctx, rw, b, b_ty);
    let c_val = registers_value(ctx, rw, c, cd_ty);

    let shape = op.shape(ctx).0;
    let Some(result) = emit_wmma(ctx, rw, form, shape, (a_val, b_val, c_val), cd_ty) else {
        return input_err!(op.loc(ctx), MatrixShapeUnsupported(shape));
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

    fn shape(m: usize, n: usize, k: usize) -> MatrixShape {
        MatrixShape { m, n, k }
    }

    #[test]
    fn a_tile_deeper_than_the_instruction_is_several_of_them() {
        assert_eq!(instruction_steps(shape(16, 16, 16)), Some(1));
        assert_eq!(instruction_steps(shape(16, 16, 32)), Some(2));
    }

    #[test]
    fn a_tile_that_does_not_divide_has_no_lowering() {
        assert_eq!(instruction_steps(shape(16, 16, 24)), None);
        assert_eq!(instruction_steps(shape(16, 16, 8)), None);
        assert_eq!(instruction_steps(shape(16, 16, 0)), None);
    }

    /// Every instruction writes a 16x16 output, so a tile of any other has no lowering.
    #[test]
    fn a_tile_of_another_output_shape_has_no_lowering() {
        assert_eq!(instruction_steps(shape(32, 8, 16)), None);
        assert_eq!(instruction_steps(shape(16, 8, 32)), None);
        assert_eq!(instruction_steps(shape(32, 32, 16)), None);
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
