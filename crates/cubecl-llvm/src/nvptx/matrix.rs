//! NVPTX matrix operations.

use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LdMatrixOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp,
    StMatrixOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use cubecl_core::ir::types::{MatrixIdent, MatrixLayout, MatrixShape};

use pliron::input_err;
use pliron::printable::Printable;
use pliron_llvm::types::{StructLayout, StructType};
use thiserror::Error;

use crate::shared::matrix::{
    registers_array_ty, registers_as_vector, registers_value, vector_into_array,
};
use crate::shared::plane::bitcast;
use crate::shared::to_llvm::prelude::*;

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
    "the NVPTX backend lowers the cooperative matrix API through `wmma`, whose fragment layout \
     is opaque; `{0}` is part of the manual `mma.sync` API, which it does not implement"
)]
pub struct MatrixManualUnsupported(&'static str);

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
}

/// Register layouts from LLVM `IntrinsicsNVVM.td`.
fn fragment_of(ctx: &Context, matrix: &MatrixType) -> Option<Fragment> {
    let elem = matrix.elem_ty;
    match matrix.ident {
        MatrixIdent::A | MatrixIdent::B if elem.is_float16(ctx) => Some(Fragment {
            regs: 8,
            per_reg: 2,
        }),
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

fn register_ty(ctx: &mut Context, frag: Fragment, elem: TypeHandle) -> TypeHandle {
    if frag.per_reg == 1 {
        elem
    } else {
        LlvmVectorType::get(ctx, elem, frag.per_reg as u32, VectorTypeKind::Fixed).into()
    }
}

fn wmma_type(ctx: &Context, elem: TypeHandle) -> Option<&'static str> {
    if elem.is_float32(ctx) {
        Some("f32")
    } else if elem.is_float16(ctx) {
        Some("f16")
    } else {
        None
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

fn extract_lane(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    i: usize,
) -> Value {
    let index = insert_i32_const(ctx, rw, i as i32);
    let op = llvm::ExtractElementOp::new(ctx, vector, index);
    insert(ctx, rw, &op)
}

fn insert_lane(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    value: Value,
    i: usize,
) -> Value {
    let index = insert_i32_const(ctx, rw, i as i32);
    let op = llvm::InsertElementOp::new(ctx, vector, value, index);
    insert(ctx, rw, &op)
}

fn poison(ctx: &mut Context, rw: &mut DialectConversionRewriter, ty: TypeHandle) -> Value {
    let op = llvm::PoisonOp::new(ctx, ty);
    insert(ctx, rw, &op)
}

fn to_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    fragment: Value,
    frag: Fragment,
    reg_ty: TypeHandle,
) -> Vec<Value> {
    (0..frag.regs)
        .map(|r| {
            if frag.per_reg == 1 {
                return extract_lane(ctx, rw, fragment, r);
            }
            let mut reg = poison(ctx, rw, reg_ty);
            for lane in 0..frag.per_reg {
                let element = extract_lane(ctx, rw, fragment, r * frag.per_reg + lane);
                reg = insert_lane(ctx, rw, reg, element, lane);
            }
            reg
        })
        .collect()
}

fn from_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    regs: &[Value],
    frag: Fragment,
    frag_ty: TypeHandle,
) -> Value {
    let mut acc = poison(ctx, rw, frag_ty);
    for (r, &reg) in regs.iter().enumerate() {
        if frag.per_reg == 1 {
            acc = insert_lane(ctx, rw, acc, reg, r);
            continue;
        }
        for lane in 0..frag.per_reg {
            let element = extract_lane(ctx, rw, reg, lane);
            acc = insert_lane(ctx, rw, acc, element, r * frag.per_reg + lane);
        }
    }
    acc
}

fn call_returning_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    reg_tys: Vec<TypeHandle>,
    args: Vec<Value>,
) -> Vec<Value> {
    let count = reg_tys.len();
    // A single return register uses a scalar type; multiple registers use a struct.
    let result_ty: TypeHandle = if count == 1 {
        reg_tys[0]
    } else {
        StructType::get_unnamed(ctx, (reg_tys, StructLayout::Unpacked)).into()
    };

    let arg_tys = args.iter().map(|arg| arg.get_type(ctx)).collect();
    let fn_ty = FuncType::get(ctx, result_ty, arg_tys, false);
    let call = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
    let result = insert(ctx, rw, &call);

    if count == 1 {
        return vec![result];
    }
    (0..count)
        .map(|field| {
            let op = llvm::ExtractValueOp::new(ctx, result, vec![field as u32])
                .expect("a constant index into the returned registers");
            insert(ctx, rw, &op)
        })
        .collect()
}

fn call_void(ctx: &mut Context, rw: &mut DialectConversionRewriter, name: &str, args: Vec<Value>) {
    let arg_tys = args.iter().map(|arg| arg.get_type(ctx)).collect();
    let void_ty = pliron_llvm::types::VoidType::get(ctx).into();
    let fn_ty = FuncType::get(ctx, void_ty, arg_tys, false);
    let call = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
    rw.insert_op(ctx, &call);
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
    let elem = cube_type_to_llvm(ctx, ty.elem_ty);
    let reg_ty = register_ty(ctx, frag, elem);
    let stride = stride_as_i32(ctx, rw, stride);

    // WMMA memory instructions require the source address space for specialization.
    let source = in_origin_space(ctx, rw, source);
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

    let value = from_registers(ctx, rw, &regs, frag, frag_ty);
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
    let elem = cube_type_to_llvm(ctx, ty.elem_ty);
    let reg_ty = register_ty(ctx, frag, elem);
    let stride = stride_as_i32(ctx, rw, stride);

    let value = load_fragment(ctx, rw, matrix, frag_ty);
    let regs = to_registers(ctx, rw, value, frag, reg_ty);

    let destination = in_origin_space(ctx, rw, destination);
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

    let (Some(ab_frag), Some(cd_frag)) = (fragment_of(ctx, &a_ty), fragment_of(ctx, &c_ty)) else {
        let culprit = if fragment_of(ctx, &a_ty).is_none() {
            a_ty.elem_ty
        } else {
            c_ty.elem_ty
        };
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

    let ab_frag_ty = fragment_ty(ctx, &a_ty);
    let cd_frag_ty = fragment_ty(ctx, &c_ty);
    let ab_elem = cube_type_to_llvm(ctx, a_ty.elem_ty);
    let cd_elem = cube_type_to_llvm(ctx, c_ty.elem_ty);
    let ab_reg_ty = register_ty(ctx, ab_frag, ab_elem);
    let cd_reg_ty = register_ty(ctx, cd_frag, cd_elem);

    let a_val = load_fragment(ctx, rw, a, ab_frag_ty);
    let b_val = load_fragment(ctx, rw, b, ab_frag_ty);
    let c_val = load_fragment(ctx, rw, c, cd_frag_ty);

    let mut args = to_registers(ctx, rw, a_val, ab_frag, ab_reg_ty);
    args.extend(to_registers(ctx, rw, b_val, ab_frag, ab_reg_ty));
    args.extend(to_registers(ctx, rw, c_val, cd_frag, cd_reg_ty));

    let name = format!(
        "llvm.nvvm.wmma.{}.mma.{a_layout}.{b_layout}.{cd_name}.{cd_name}",
        geometry(a_ty.shape),
    );
    let regs = call_returning_registers(ctx, rw, &name, vec![cd_reg_ty; cd_frag.regs], args);

    let result = from_registers(ctx, rw, &regs, cd_frag, cd_frag_ty);
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
    debug_assert_eq!(
        in_frag.elems(),
        out_frag.elems(),
        "a cast keeps the element count and changes only their width"
    );

    let in_frag_ty = fragment_ty(ctx, &in_ty);
    let out_frag_ty = fragment_ty(ctx, &out_ty);
    let value = load_fragment(ctx, rw, input, in_frag_ty);

    let in_bits = in_ty.elem_ty.size_bits(ctx);
    let out_bits = out_ty.elem_ty.size_bits(ctx);
    let result = if in_bits > out_bits {
        let cast = llvm::FPTruncOp::new(ctx, value, out_frag_ty);
        cast.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        insert(ctx, rw, &cast)
    } else if in_bits < out_bits {
        let cast = llvm::FPExtOp::new(ctx, value, out_frag_ty);
        cast.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        insert(ctx, rw, &cast)
    } else {
        debug_assert_eq!(
            cube_type_to_llvm(ctx, in_ty.elem_ty),
            cube_type_to_llvm(ctx, out_ty.elem_ty),
            "a cast of the same width between two distinct LLVM types needs a conversion"
        );
        value
    };

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

/// Register packing for manual matrix operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegisterForm {
    /// Several elements per register.
    Packed(usize),
    /// One element per register.
    Scalar,
    /// Elements packed into an opaque 32-bit register.
    Word,
}

fn mma_type(ctx: &Context, elem: TypeHandle) -> Option<(&'static str, RegisterForm)> {
    if elem.is_float16(ctx) {
        Some(("f16", RegisterForm::Packed(2)))
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

fn word_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

fn registers_of(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    vector: Value,
    form: RegisterForm,
) -> Vec<Value> {
    let (elems, elem) = {
        let ty = vector.get_type(ctx);
        let ty = ty.deref(ctx);
        let vec = ty
            .downcast_ref::<LlvmVectorType>()
            .expect("a fragment is held in a vector");
        (vec.num_elements() as usize, vec.elem_type())
    };

    match form {
        RegisterForm::Scalar => (0..elems)
            .map(|i| extract_lane(ctx, rw, vector, i))
            .collect(),
        RegisterForm::Packed(per_reg) => {
            let reg_ty =
                LlvmVectorType::get(ctx, elem, per_reg as u32, VectorTypeKind::Fixed).into();
            let frag = Fragment {
                regs: elems / per_reg,
                per_reg,
            };
            to_registers(ctx, rw, vector, frag, reg_ty)
        }
        RegisterForm::Word => {
            let word = word_ty(ctx);
            let bits = elems * elem.size_bits(ctx);
            let words = bits / 32;
            let words_ty =
                LlvmVectorType::get(ctx, word, words as u32, VectorTypeKind::Fixed).into();
            let as_words = bitcast(ctx, rw, vector, words_ty);
            (0..words)
                .map(|i| extract_lane(ctx, rw, as_words, i))
                .collect()
        }
    }
}

fn registers_into(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    regs: &[Value],
    vector_ty: TypeHandle,
    form: RegisterForm,
) -> Value {
    let (elems, elem) = {
        let ty = vector_ty.deref(ctx);
        let vec = ty
            .downcast_ref::<LlvmVectorType>()
            .expect("a fragment is held in a vector");
        (vec.num_elements() as usize, vec.elem_type())
    };

    match form {
        RegisterForm::Scalar => {
            let mut acc = poison(ctx, rw, vector_ty);
            for (i, &reg) in regs.iter().enumerate() {
                acc = insert_lane(ctx, rw, acc, reg, i);
            }
            acc
        }
        RegisterForm::Packed(per_reg) => {
            let frag = Fragment {
                regs: elems / per_reg,
                per_reg,
            };
            from_registers(ctx, rw, regs, frag, vector_ty)
        }
        RegisterForm::Word => {
            let word = word_ty(ctx);
            let words_ty =
                LlvmVectorType::get(ctx, word, regs.len() as u32, VectorTypeKind::Fixed).into();
            let mut acc = poison(ctx, rw, words_ty);
            for (i, &reg) in regs.iter().enumerate() {
                acc = insert_lane(ctx, rw, acc, reg, i);
            }
            let _ = elem;
            bitcast(ctx, rw, acc, vector_ty)
        }
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

    let mut args = registers_of(ctx, rw, a_val, a_form);
    args.extend(registers_of(ctx, rw, b_val, b_form));
    let c_regs = registers_of(ctx, rw, c_val, cd_form);
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

    let result = registers_into(ctx, rw, &regs, cd_vec_ty, cd_form);
    let d_array_ty = registers_array_ty(ctx, operands_info, d);
    let result = vector_into_array(ctx, rw, result, d_array_ty);
    store_fragment(ctx, rw, d, result);

    rw.erase_operation(ctx, old_op);
    Ok(())
}

/// Matrix tile instructions require shared memory.
const SHARED_ADDRESS_SPACE: u32 = 3;

const GLOBAL_ADDRESS_SPACE: u32 = 1;

const GENERIC_ADDRESS_SPACE: u32 = 0;

fn address_space(ctx: &Context, value: Value) -> Option<u32> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmPointerType>()
        .map(LlvmPointerType::address_space)
}

/// Known source address space, or `None` when the origin is unknown.
fn origin_address_space(ctx: &Context, ptr: Value) -> Option<u32> {
    let mut ptr = ptr;
    loop {
        match address_space(ctx, ptr) {
            Some(GENERIC_ADDRESS_SPACE) => {}
            space => return space,
        }
        let op = ptr.defining_op()?;
        let derives_from_its_pointer = Operation::get_op::<llvm::GetElementPtrOp>(op, ctx)
            .is_some()
            || Operation::get_op::<llvm::AddrSpaceCastOp>(op, ctx).is_some();
        if !derives_from_its_pointer {
            return None;
        }
        ptr = op.deref(ctx).get_operand(0);
    }
}

fn in_origin_space(ctx: &mut Context, rw: &mut DialectConversionRewriter, ptr: Value) -> Value {
    match origin_address_space(ctx, ptr) {
        Some(space @ (SHARED_ADDRESS_SPACE | GLOBAL_ADDRESS_SPACE)) => {
            let ty: TypeHandle = LlvmPointerType::get(ctx, space).into();
            if ptr.get_type(ctx) == ty {
                return ptr;
            }
            let op = llvm::AddrSpaceCastOp::new(ctx, ptr, ty);
            insert(ctx, rw, &op)
        }
        _ => ptr,
    }
}

fn as_shared(ctx: &mut Context, rw: &mut DialectConversionRewriter, ptr: Value) -> Value {
    let shared_ty: TypeHandle = LlvmPointerType::get(ctx, SHARED_ADDRESS_SPACE).into();
    if ptr.get_type(ctx) == shared_ty {
        return ptr;
    }
    let op = llvm::AddrSpaceCastOp::new(ctx, ptr, shared_ty);
    insert(ctx, rw, &op)
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
    let source = as_shared(ctx, rw, ptr);
    let word = word_ty(ctx);

    let name = format!(
        "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x{factor}{}.b16.{}",
        transpose_name(transpose),
        llvm_mangled_ty(ctx, source.get_type(ctx)),
    );
    let regs = call_returning_registers(ctx, rw, &name, vec![word; factor], vec![source]);

    let value = registers_into(ctx, rw, &regs, out_vec_ty, RegisterForm::Word);
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
    let regs = registers_of(ctx, rw, value, RegisterForm::Word);
    let target = as_shared(ctx, rw, destination);

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
    use cubecl_core::ir::types::scalar::{Float16Type, Float32Type};

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

    #[test]
    fn a_fragment_no_instruction_takes_is_refused() {
        let ctx = Context::default();
        let f32: TypeHandle = Float32Type::get(&ctx).into();

        assert_eq!(fragment_of(&ctx, &matrix(MatrixIdent::A, f32)), None);
        assert_eq!(fragment_of(&ctx, &matrix(MatrixIdent::B, f32)), None);
    }

    #[test]
    fn the_fragments_own_layout_wins_over_the_accesss() {
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
