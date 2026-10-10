//! NVPTX matrix operations.

use crate::{
    nvptx::{
        address::NvptxSpace,
        registers::{
            RegisterForm, call_returning_registers, extract_lane, insert_lane, load_fragment,
            poison, store_fragment,
        },
    },
    prelude::*,
    shared::{
        matrix::{
            convert_lanes, registers_array_ty, registers_as_vector, registers_value,
            vector_into_array,
        },
        plane::bitcast,
    },
};
use cubecl_core::ir::{
    dialect::matrix::{
        CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp,
        RowIndexOp, StMatrixOp, StoreOp,
    },
    types::{MatrixIdent, MatrixLayout, MatrixShape, matrix::MatrixType},
};

#[derive(Debug, Error)]
#[error("no WMMA instruction takes a {0} fragment element on this target")]
pub struct MatrixElemUnsupported(String);

#[derive(Debug, Error)]
#[error(
    "casting a {0} fragment to a {1} one needs a relayout the NVPTX lowering cannot do: a WMMA \
     fragment's layout is opaque, so the only way between two of them is through memory"
)]
pub struct MatrixRelayoutUnsupported(MatrixIdent, MatrixIdent);

#[derive(Debug, Error)]
#[error(
    "casting a {ident} fragment of {from} to {to} needs a relayout the NVPTX lowering cannot do: \
     the {from} fragment holds {from_elems} elements per lane and the {to} one {to_elems}, so \
     the two hold different elements; convert through memory instead"
)]
pub struct MatrixElementLayoutUnsupported {
    ident: MatrixIdent,
    from: String,
    from_elems: usize,
    to: String,
    to_elems: usize,
}

#[derive(Debug, Error)]
#[error(
    "the NVPTX backend lowers the cooperative matrix API through `wmma`, whose fragment layout \
     is opaque; `{0}` is part of the manual `mma.sync` API, which it does not implement"
)]
pub struct MatrixManualUnsupported(&'static str);

/// How a WMMA fragment is laid out in registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fragment {
    regs: usize,
    /// Scalar elements per register.
    per_reg: usize,
}

impl Fragment {
    /// Scalar elements per lane, including duplicated input elements.
    fn elems(&self) -> usize {
        self.regs * self.per_reg
    }

    /// How the instructions take the fragment's registers: `tf32` and `bf16` as opaque words,
    /// the other types as themselves, packed when a register holds several.
    fn register_form(&self, ctx: &Context, elem: TypeHandle) -> RegisterForm {
        if elem.is_tfloat32(ctx) || elem.is_bfloat16(ctx) {
            RegisterForm::Word
        } else if self.per_reg == 1 {
            RegisterForm::Scalar
        } else {
            RegisterForm::Packed(self.per_reg)
        }
    }

    /// The type of one register of the fragment.
    fn register_ty(&self, ctx: &mut Context, elem: TypeHandle) -> TypeHandle {
        let form = self.register_form(ctx, elem);
        let elem = cube_type_to_llvm(ctx, elem);
        form.register_ty(ctx, elem)
    }
}

/// Register layouts from LLVM `IntrinsicsNVVM.td`.
fn fragment_of(ctx: &Context, matrix: &MatrixType) -> Option<Fragment> {
    let elem = matrix.elem_ty;
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B if elem.is_tfloat32(ctx) => Some(Fragment {
            regs: 4,
            per_reg: 1,
        }),
        MatrixIdent::A | MatrixIdent::B if elem.is_float16(ctx) => Some(Fragment {
            regs: 8,
            per_reg: 2,
        }),
        // Unlike `f16`, a `bf16` fragment holds each element once, so its size follows the
        // tile: two to an `i32`, `rows * k / 32` per lane.
        MatrixIdent::A | MatrixIdent::B if elem.is_bfloat16(ctx) => {
            let MatrixShape { m, n, k } = matrix.shape;
            let rows = if matrix.ident == MatrixIdent::A { m } else { n };
            Some(Fragment {
                regs: rows * k / 32 / 2,
                per_reg: 2,
            })
        }
        MatrixIdent::Accumulator if elem.is_float16(ctx) => Some(Fragment {
            regs: 4,
            per_reg: 2,
        }),
        MatrixIdent::Accumulator if elem.is_float32(ctx) => Some(Fragment {
            regs: 8,
            per_reg: 1,
        }),
        _ => None,
    }
}

pub(crate) fn fragment_ty(ctx: &Context, matrix: &MatrixType) -> TypeHandle {
    let elems = fragment_of(ctx, matrix).map_or(1, |frag| frag.elems());
    let elem = cube_type_to_llvm(ctx, matrix.elem_ty);
    LlvmVectorType::get(ctx, elem, elems as u32, VectorTypeKind::Fixed).into()
}

fn wmma_type(ctx: &Context, elem: TypeHandle) -> Option<&'static str> {
    if elem.is_tfloat32(ctx) {
        Some("tf32")
    } else if elem.is_float32(ctx) {
        Some("f32")
    } else if elem.is_float16(ctx) {
        Some("f16")
    } else if elem.is_bfloat16(ctx) {
        Some("bf16")
    } else {
        None
    }
}

/// Round FP32 storage to TF32 using the same ties-away rounding as CUDA `__float_to_tf32`.
/// NVVM returns the encoding in an i32 register; keep FP32 storage outside matrix calls.
pub(crate) fn round_tf32(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
) -> Value {
    let ty = value.get_type(ctx);
    let lanes = ty
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|ty| ty.num_elements());
    if let Some(lanes) = lanes {
        let mut rounded = poison(ctx, rw, ty);
        for lane in 0..lanes as usize {
            let scalar = extract_lane(ctx, rw, value, lane);
            let scalar = round_tf32(ctx, rw, scalar);
            rounded = insert_lane(ctx, rw, rounded, scalar, lane);
        }
        rounded
    } else {
        let word = i32_ty(ctx);
        let op = call_op(ctx, "llvm.nvvm.f2tf32.rna", word, vec![value]);
        let bits = insert(ctx, rw, &op);
        bitcast(ctx, rw, bits, ty)
    }
}

fn geometry(shape: MatrixShape) -> String {
    let MatrixShape { m, n, k } = shape;
    format!("m{m}n{n}k{k}")
}

fn layout_name(layout: MatrixLayout) -> Option<&'static str> {
    match layout {
        MatrixLayout::RowMajor => Some("row"),
        MatrixLayout::ColMajor => Some("col"),
        MatrixLayout::Undefined => None,
    }
}

/// Declared fragment layouts take precedence over access layouts.
fn access_layout(matrix: &MatrixType, op_layout: MatrixLayout) -> Option<&'static str> {
    layout_name(matrix.layout).or_else(|| layout_name(op_layout))
}

fn matrix_of(ctx: &Context, info: &OperandsInfo, value: Value) -> MatrixType {
    let pointee = cube_pointee(ctx, info, value, |pointee| {
        pointee.downcast_ref::<MatrixType>().copied()
    });
    pointee.expect("a matrix operand points at a matrix")
}

/// The registers a WMMA instruction takes for the fragment `matrix` points at.
fn fragment_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    matrix: Value,
    ty: &MatrixType,
    frag: Fragment,
) -> Vec<Value> {
    let frag_ty = fragment_ty(ctx, ty);
    let value = load_fragment(ctx, rw, matrix, frag_ty);
    frag.register_form(ctx, ty.elem_ty).split(ctx, rw, value)
}

fn stride_as_i32(ctx: &mut Context, rw: &mut DialectConversionRewriter, stride: Value) -> Value {
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let ty = stride.get_type(ctx);
    if ty == i32_ty {
        return stride;
    }
    let width = {
        let ty = ty.deref(ctx);
        ty.downcast_ref::<IntegerType>()
            .map(|int| int.width())
            .expect("a matrix stride is an integer")
    };
    let op: Ptr<Operation> = if width > 32 {
        llvm::TruncOp::new(ctx, stride, i32_ty).get_operation()
    } else {
        llvm::ZExtOp::new_with_nneg(ctx, stride, i32_ty, false).get_operation()
    };
    rw.insert_operation(ctx, op);
    op.deref(ctx).get_result(0)
}

fn unsupported_elem(ctx: &Context, elem: TypeHandle) -> MatrixElemUnsupported {
    MatrixElemUnsupported(elem.disp(ctx).to_string())
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
    let Some(frag) = fragment_of(ctx, &ty) else {
        return input_err!(op.loc(ctx), unsupported_elem(ctx, ty.elem_ty));
    };
    let frag_ty = fragment_ty(ctx, &ty);

    let filled = insert_splat(ctx, rw, frag_ty, value, frag.elems());
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
    let op_layout = op.layout(ctx).0;

    let ty = matrix_of(ctx, operands_info, matrix);
    let (Some(frag), Some(elem_name)) = (fragment_of(ctx, &ty), wmma_type(ctx, ty.elem_ty)) else {
        return input_err!(op.loc(ctx), unsupported_elem(ctx, ty.elem_ty));
    };
    let Some(layout) = access_layout(&ty, op_layout) else {
        return input_err!(op.loc(ctx), MatrixLayoutUnknown(ty.ident));
    };

    let frag_ty = fragment_ty(ctx, &ty);
    let reg_ty = frag.register_ty(ctx, ty.elem_ty);
    let stride = stride_as_i32(ctx, rw, stride);

    // WMMA memory instructions require the source address space for specialization.
    let source = NvptxSpace::narrow(ctx, rw, source);
    let name = format!(
        "llvm.nvvm.wmma.{}.load.{}.{layout}.stride.{elem_name}.{}",
        geometry(ty.shape),
        fragment_name(ty.ident),
        llvm_mangled_ty(ctx, source.get_type(ctx)),
    );
    let regs = call_returning_registers(
        ctx,
        rw,
        &name,
        vec![reg_ty; frag.regs],
        vec![source, stride],
    );

    let value = frag
        .register_form(ctx, ty.elem_ty)
        .join(ctx, rw, &regs, frag_ty);
    store_fragment(ctx, rw, matrix, value);

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
    let op_layout = op.layout(ctx).0;

    let ty = matrix_of(ctx, operands_info, matrix);
    let (Some(frag), Some(elem_name)) = (fragment_of(ctx, &ty), wmma_type(ctx, ty.elem_ty)) else {
        return input_err!(op.loc(ctx), unsupported_elem(ctx, ty.elem_ty));
    };
    let Some(layout) = layout_name(op_layout).or_else(|| layout_name(ty.layout)) else {
        return input_err!(op.loc(ctx), MatrixLayoutUnknown(ty.ident));
    };

    let frag_ty = fragment_ty(ctx, &ty);
    let stride = stride_as_i32(ctx, rw, stride);

    let value = load_fragment(ctx, rw, matrix, frag_ty);
    let regs = frag.register_form(ctx, ty.elem_ty).split(ctx, rw, value);

    let destination = NvptxSpace::narrow(ctx, rw, destination);
    let name = format!(
        "llvm.nvvm.wmma.{}.store.d.{layout}.stride.{elem_name}.{}",
        geometry(ty.shape),
        llvm_mangled_ty(ctx, destination.get_type(ctx)),
    );
    let mut args = vec![destination];
    args.extend(regs);
    args.push(stride);
    call_void(ctx, rw, &name, args);

    rw.erase_operation(ctx, old_op);
    Ok(())
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

    // A and B are separate fragments: a `bf16` tile that is not square sizes them differently.
    let (Some(a_frag), Some(b_frag), Some(cd_frag)) = (
        fragment_of(ctx, &a_ty),
        fragment_of(ctx, &b_ty),
        fragment_of(ctx, &c_ty),
    ) else {
        let culprit = [&a_ty, &b_ty, &c_ty]
            .into_iter()
            .find(|ty| fragment_of(ctx, ty).is_none())
            .expect("one of the three has no fragment")
            .elem_ty;
        return input_err!(op.loc(ctx), unsupported_elem(ctx, culprit));
    };
    let Some(cd_name) = wmma_type(ctx, c_ty.elem_ty) else {
        return input_err!(op.loc(ctx), unsupported_elem(ctx, c_ty.elem_ty));
    };
    // Operand layouts must be known at compile time.
    let (Some(a_layout), Some(b_layout)) = (layout_name(a_ty.layout), layout_name(b_ty.layout))
    else {
        let culprit = if layout_name(a_ty.layout).is_none() {
            MatrixIdent::A
        } else {
            MatrixIdent::B
        };
        return input_err!(op.loc(ctx), MatrixLayoutUnknown(culprit));
    };

    let cd_frag_ty = fragment_ty(ctx, &c_ty);
    let cd_form = cd_frag.register_form(ctx, c_ty.elem_ty);
    let cd_reg_ty = cd_frag.register_ty(ctx, c_ty.elem_ty);

    let mut args = fragment_registers(ctx, rw, a, &a_ty, a_frag);
    args.extend(fragment_registers(ctx, rw, b, &b_ty, b_frag));
    let c_val = load_fragment(ctx, rw, c, cd_frag_ty);
    args.extend(cd_form.split(ctx, rw, c_val));

    let name = format!(
        "llvm.nvvm.wmma.{}.mma.{a_layout}.{b_layout}.{}",
        geometry(a_ty.shape),
        mma_signature(
            wmma_type(ctx, a_ty.elem_ty).unwrap(),
            wmma_type(ctx, b_ty.elem_ty).unwrap(),
            cd_name
        ),
    );
    let regs = call_returning_registers(ctx, rw, &name, vec![cd_reg_ty; cd_frag.regs], args);

    let result = cd_form.join(ctx, rw, &regs, cd_frag_ty);
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

    // Casts between fragment kinds require a different lane layout.
    if in_ty.ident != out_ty.ident {
        return input_err!(
            op.loc(ctx),
            MatrixRelayoutUnsupported(in_ty.ident, out_ty.ident)
        );
    }

    let (Some(in_frag), Some(out_frag)) = (fragment_of(ctx, &in_ty), fragment_of(ctx, &out_ty))
    else {
        let culprit = if fragment_of(ctx, &in_ty).is_none() {
            in_ty.elem_ty
        } else {
            out_ty.elem_ty
        };
        return input_err!(op.loc(ctx), unsupported_elem(ctx, culprit));
    };
    if in_frag.elems() != out_frag.elems() {
        return input_err!(
            op.loc(ctx),
            MatrixElementLayoutUnsupported {
                ident: in_ty.ident,
                from: in_ty.elem_ty.disp(ctx).to_string(),
                from_elems: in_frag.elems(),
                to: out_ty.elem_ty.disp(ctx).to_string(),
                to_elems: out_frag.elems(),
            }
        );
    }

    let in_frag_ty = fragment_ty(ctx, &in_ty);
    let value = load_fragment(ctx, rw, input, in_frag_ty);
    let result = convert_lanes(ctx, rw, value, out_ty.elem_ty);
    store_fragment(ctx, rw, output, result);
    rw.erase_operation(ctx, old_op);
    Ok(())
}

#[derive(Debug, Error)]
#[error(
    "the {0} fragment has no layout, and a WMMA instruction names the layout of what it reads; \
     declare the fragment row- or column-major, or give the access one"
)]
pub struct MatrixLayoutUnknown(MatrixIdent);

fn fragment_name(ident: MatrixIdent) -> &'static str {
    match ident {
        MatrixIdent::A => "a",
        MatrixIdent::B => "b",
        MatrixIdent::Accumulator => "c",
    }
}

fn mma_type(ctx: &Context, elem: TypeHandle) -> Option<(&'static str, RegisterForm)> {
    if elem.is_tfloat32(ctx) {
        Some(("tf32", RegisterForm::Word))
    } else if elem.is_float16(ctx) {
        Some(("f16", RegisterForm::Packed(2)))
    } else if elem.is_bfloat16(ctx) {
        Some(("bf16", RegisterForm::Word))
    } else if elem.is_float32(ctx) {
        Some(("f32", RegisterForm::Scalar))
    } else if elem.is_int(ctx) && elem.size_bits(ctx) == 8 {
        let name = if elem.is_signed_int(ctx) { "s8" } else { "u8" };
        Some((name, RegisterForm::Word))
    } else if elem.is_int(ctx) && elem.size_bits(ctx) == 32 {
        Some(("s32", RegisterForm::Scalar))
    } else {
        None
    }
}

/// Type suffixes follow `MMA_SIGNATURE` in LLVM `IntrinsicsNVVM.td`.
fn mma_signature(a: &str, b: &str, cd: &str) -> String {
    if a == "f16" {
        format!("{cd}.{cd}")
    } else if a != b {
        format!("{a}.{b}")
    } else {
        a.to_string()
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

    let (a_vec_ty, a_elem) = registers_as_vector(ctx, operands_info, a);
    let (b_vec_ty, b_elem) = registers_as_vector(ctx, operands_info, b);
    let (cd_vec_ty, cd_elem) = registers_as_vector(ctx, operands_info, c);

    let (Some((a_name, a_form)), Some((b_name, b_form)), Some((cd_name, cd_form))) = (
        mma_type(ctx, a_elem),
        mma_type(ctx, b_elem),
        mma_type(ctx, cd_elem),
    ) else {
        let culprit = [a_elem, b_elem, cd_elem]
            .into_iter()
            .find(|&elem| mma_type(ctx, elem).is_none())
            .expect("one of the three has no form");
        return input_err!(op.loc(ctx), unsupported_elem(ctx, culprit));
    };

    let a_val = registers_value(ctx, rw, a, a_vec_ty);
    let b_val = registers_value(ctx, rw, b, b_vec_ty);
    let c_val = registers_value(ctx, rw, c, cd_vec_ty);

    let mut args = a_form.split(ctx, rw, a_val);
    args.extend(b_form.split(ctx, rw, b_val));
    let c_regs = cd_form.split(ctx, rw, c_val);
    let result_count = c_regs.len();
    let reg_tys: Vec<TypeHandle> = c_regs.iter().map(|reg| reg.get_type(ctx)).collect();
    args.extend(c_regs);

    // These MMA shapes require row-major A and column-major B.
    let MatrixShape { m, n, k } = *op.shape(ctx).clone();
    let name = format!(
        "llvm.nvvm.mma.m{m}n{n}k{k}.row.col.{}",
        mma_signature(a_name, b_name, cd_name),
    );
    let regs = call_returning_registers(ctx, rw, &name, reg_tys, args);
    debug_assert_eq!(regs.len(), result_count);

    let result = cd_form.join(ctx, rw, &regs, cd_vec_ty);
    let d_array_ty = registers_array_ty(ctx, operands_info, d);
    let result = vector_into_array(ctx, rw, result, d_array_ty);
    store_fragment(ctx, rw, d, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

fn transpose_name(transpose: bool) -> &'static str {
    if transpose { ".trans" } else { "" }
}

pub(crate) fn ld_matrix(
    op: &LdMatrixOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let ptr = op.ptr(ctx);
    let out_arr = op.out_arr(ctx);
    let factor = op.factor(ctx).0;
    let transpose = op.transpose(ctx).0;

    let (out_vec_ty, _) = registers_as_vector(ctx, operands_info, out_arr);
    let source = NvptxSpace::Shared.cast(ctx, rw, ptr);
    let word = i32_ty(ctx);

    let name = format!(
        "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x{factor}{}.b16.{}",
        transpose_name(transpose),
        llvm_mangled_ty(ctx, source.get_type(ctx)),
    );
    let regs = call_returning_registers(ctx, rw, &name, vec![word; factor], vec![source]);

    let value = RegisterForm::Word.join(ctx, rw, &regs, out_vec_ty);
    let out_array_ty = registers_array_ty(ctx, operands_info, out_arr);
    let value = vector_into_array(ctx, rw, value, out_array_ty);
    store_fragment(ctx, rw, out_arr, value);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

pub(crate) fn st_matrix(
    op: &StMatrixOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let old_op = op.get_operation();
    let registers = op.registers(ctx);
    let destination = op.destination(ctx);
    let factor = op.factor(ctx).0;
    let transpose = op.transpose(ctx).0;

    let (vec_ty, _) = registers_as_vector(ctx, operands_info, registers);
    let value = registers_value(ctx, rw, registers, vec_ty);
    let regs = RegisterForm::Word.split(ctx, rw, value);
    let target = NvptxSpace::Shared.cast(ctx, rw, destination);

    let name = format!(
        "llvm.nvvm.stmatrix.sync.aligned.m8n8.x{factor}{}.b16.{}",
        transpose_name(transpose),
        llvm_mangled_ty(ctx, target.get_type(ctx)),
    );
    let mut args = vec![target];
    args.extend(regs);
    call_void(ctx, rw, &name, args);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// Matrix indices must be lowered before dialect conversion.
macro_rules! lowered_by_the_polyfill {
    ($fn_name:ident, $cube_op:ty) => {
        pub(crate) fn $fn_name(
            _op: &$cube_op,
            _ctx: &mut Context,
            _rw: &mut DialectConversionRewriter,
            _operands_info: &OperandsInfo,
        ) -> Result<()> {
            unreachable!(
                "`{}` is expanded by `LowerComplexOpPass` on the NVPTX target, which runs \
                 before this conversion",
                stringify!($cube_op)
            )
        }
    };
}

lowered_by_the_polyfill!(row_index, RowIndexOp);
lowered_by_the_polyfill!(col_index, ColIndexOp);

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_core::ir::types::MatrixScope;
    use cubecl_core::ir::types::scalar::{BFloat16Type, Float16Type, Float32Type};

    fn matrix(ident: MatrixIdent, elem_ty: TypeHandle) -> MatrixType {
        MatrixType {
            ident,
            shape: MatrixShape {
                m: 16,
                n: 16,
                k: 16,
            },
            elem_ty,
            layout: MatrixLayout::RowMajor,
            scope: MatrixScope::Plane,
        }
    }

    #[test]
    fn the_fragments_are_the_shapes_the_intrinsics_declare() {
        let ctx = Context::default();
        let f16: TypeHandle = Float16Type::get(&ctx).into();
        let f32: TypeHandle = Float32Type::get(&ctx).into();

        for ident in [MatrixIdent::A, MatrixIdent::B] {
            let frag = fragment_of(&ctx, &matrix(ident, f16)).unwrap();
            assert_eq!(
                frag,
                Fragment {
                    regs: 8,
                    per_reg: 2
                },
                "{ident:?}"
            );
            assert_eq!(frag.elems(), 16);
        }

        let acc16 = fragment_of(&ctx, &matrix(MatrixIdent::Accumulator, f16)).unwrap();
        let acc32 = fragment_of(&ctx, &matrix(MatrixIdent::Accumulator, f32)).unwrap();
        assert_eq!(
            acc16,
            Fragment {
                regs: 4,
                per_reg: 2
            }
        );
        assert_eq!(
            acc32,
            Fragment {
                regs: 8,
                per_reg: 1
            }
        );
        assert_eq!(acc16.elems(), acc32.elems());
    }

    /// `bf16` fragments hold each element once, two to an `i32`, so their register count
    /// follows the tile as `IntrinsicsNVVM.td` declares it.
    #[test]
    fn bf16_fragments_follow_the_tile() {
        let ctx = Context::default();
        let bf16: TypeHandle = BFloat16Type::get(&ctx).into();
        let regs = |ident, m, n| {
            let mut matrix = matrix(ident, bf16);
            matrix.shape = MatrixShape { m, n, k: 16 };
            fragment_of(&ctx, &matrix).unwrap().regs
        };
        assert_eq!(regs(MatrixIdent::A, 16, 16), 4);
        assert_eq!(regs(MatrixIdent::B, 16, 16), 4);
        assert_eq!(regs(MatrixIdent::A, 32, 8), 8);
        assert_eq!(regs(MatrixIdent::B, 32, 8), 2);
        assert_eq!(regs(MatrixIdent::A, 8, 32), 2);
        assert_eq!(regs(MatrixIdent::B, 8, 32), 8);
        assert_eq!(
            fragment_of(&ctx, &matrix(MatrixIdent::Accumulator, bf16)),
            None
        );
    }

    #[test]
    fn a_fragment_no_instruction_takes_is_refused() {
        let ctx = Context::default();
        let f32: TypeHandle = Float32Type::get(&ctx).into();

        assert_eq!(fragment_of(&ctx, &matrix(MatrixIdent::A, f32)), None);
        assert_eq!(fragment_of(&ctx, &matrix(MatrixIdent::B, f32)), None);
    }

    #[test]
    fn the_fragments_own_layout_wins_over_the_access() {
        let ctx = Context::default();
        let f32: TypeHandle = Float32Type::get(&ctx).into();

        let a = matrix(MatrixIdent::A, f32);
        assert_eq!(access_layout(&a, MatrixLayout::ColMajor), Some("row"));

        let mut acc = matrix(MatrixIdent::Accumulator, f32);
        acc.layout = MatrixLayout::Undefined;
        assert_eq!(access_layout(&acc, MatrixLayout::ColMajor), Some("col"));
        assert_eq!(access_layout(&acc, MatrixLayout::Undefined), None);
    }

    #[test]
    fn a_geometry_is_named_for_its_tile() {
        assert_eq!(
            geometry(MatrixShape {
                m: 16,
                n: 16,
                k: 16
            }),
            "m16n16k16"
        );
        assert_eq!(geometry(MatrixShape { m: 32, n: 8, k: 16 }), "m32n8k16");
    }
}
