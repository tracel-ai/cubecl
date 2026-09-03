//! Lowering of the matrix operations to the warp's WMMA instructions.
//!
//! Where the AMDGPU lowering addresses a fragment element by element -- it knows which lane
//! holds which element of the tile, and a load is arithmetic on the tile's stride -- NVIDIA's
//! WMMA fragment is deliberately opaque. Only `wmma.load` and `wmma.store` know where anything
//! sits, and the layout is not the same between two architectures, so the whole of what this
//! module does is move fragments between an `alloca` and those instructions.
//!
//! A fragment is held here as one LLVM vector of the elements a lane owns, the same as on the
//! AMD side, because that is what lets a matrix be an `alloca` the other ops load and store.
//! The instructions want it as a list of 32 bit registers instead, so every call is bracketed
//! by [`to_registers`] and [`from_registers`]; the two are inverses and LLVM folds the
//! extract/insert chains away.
//!
//! The register counts come from LLVM's own `IntrinsicsNVVM.td`, which is what the backend
//! type-checks the calls against. They are the same for all three WMMA geometries -- the
//! tile's shape changes what an instruction computes, not how much of it a lane holds -- which
//! is why the table below is keyed only on the fragment and its element type.

use cubecl_core::ir::dialect::matrix::{
    CastOp, ColIndexOp, FillOp, LoadOp, MmaManualOp, MultiplyAccumulateOp, RowIndexOp, StoreOp,
};
use cubecl_core::ir::types::matrix::MatrixType;
use cubecl_core::ir::types::{MatrixIdent, MatrixLayout, MatrixShape};

use pliron::input_err;
use pliron::printable::Printable;
use pliron_llvm::types::{StructLayout, StructType};
use thiserror::Error;

use crate::shared::to_llvm::prelude::*;

/// A fragment element type no WMMA instruction takes.
#[derive(Debug, Error)]
#[error("no WMMA instruction takes a {0} fragment element on this target")]
pub struct MatrixElemUnsupported(String);

/// A cast that would have to move elements between lanes, which this lowering cannot do.
#[derive(Debug, Error)]
#[error(
    "casting a {0} fragment to a {1} one needs a relayout the NVPTX lowering cannot do: a WMMA \
     fragment's layout is opaque, so the only way between two of them is through memory"
)]
pub struct MatrixRelayoutUnsupported(MatrixIdent, MatrixIdent);

/// An operation the manual `mma.sync` API would answer, reached on a target that does not
/// advertise it.
#[derive(Debug, Error)]
#[error(
    "the NVPTX backend lowers the cooperative matrix API through `wmma`, whose fragment layout \
     is opaque; `{0}` is part of the manual `mma.sync` API, which it does not implement"
)]
pub struct MatrixManualUnsupported(&'static str);

/// How a fragment is held in registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fragment {
    /// Registers the instruction takes or gives.
    regs: usize,
    /// Scalar elements packed into one register: two for the 16 bit types, one for `f32`.
    per_reg: usize,
}

impl Fragment {
    /// Scalar elements of the tile this lane holds.
    ///
    /// For A and B this is twice what the tile divided by the warp would suggest: a WMMA A
    /// fragment is spread over 16 lanes rather than 32, so each element is held by two of
    /// them. That redundancy is the instruction's own, and nothing here has to account for it
    /// beyond sizing the vector.
    fn elems(&self) -> usize {
        self.regs * self.per_reg
    }
}

/// How `matrix` is held, or `None` for an element type no WMMA instruction takes.
///
/// From `IntrinsicsNVVM.td`: for the WMMA geometries, `a:f16` and `b:f16` are eight `<2 x
/// half>`, `c:f16` and `d:f16` four of them, and `c:f32` and `d:f32` eight `float`.
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

/// The LLVM vector a fragment of `matrix` lives in.
///
/// Falls back to a single element for a type with no WMMA form, which keeps the type
/// conversion total -- it has no way to report an error -- while every op below refuses the
/// same fragment with [`MatrixElemUnsupported`] before it can be used for anything.
pub(crate) fn fragment_ty(ctx: &Context, matrix: &MatrixType) -> TypeHandle {
    let elems = fragment_of(ctx, matrix).map_or(1, |frag| frag.elems());
    let elem = cube_type_to_llvm(ctx, matrix.elem_ty);
    LlvmVectorType::get(ctx, elem, elems as u32, VectorTypeKind::Fixed).into()
}

/// The register type of `frag`, whose elements are `elem`.
fn register_ty(ctx: &mut Context, frag: Fragment, elem: TypeHandle) -> TypeHandle {
    if frag.per_reg == 1 {
        elem
    } else {
        LlvmVectorType::get(ctx, elem, frag.per_reg as u32, VectorTypeKind::Fixed).into()
    }
}

/// The PTX element-type name a fragment of `elem` is called by, or `None` for one no
/// instruction takes.
fn wmma_type(ctx: &Context, elem: TypeHandle) -> Option<&'static str> {
    if elem.is_float32(ctx) {
        Some("f32")
    } else if elem.is_float16(ctx) {
        Some("f16")
    } else {
        None
    }
}

/// `m16n16k16` and the like, which is how an instruction names the tile it computes.
fn geometry(shape: MatrixShape) -> String {
    let MatrixShape { m, n, k } = shape;
    format!("m{m}n{n}k{k}")
}

/// The name a layout goes by in an instruction.
fn layout_name(layout: MatrixLayout) -> Option<&'static str> {
    match layout {
        MatrixLayout::RowMajor => Some("row"),
        MatrixLayout::ColMajor => Some("col"),
        MatrixLayout::Undefined => None,
    }
}

/// The layout to load or store `matrix` by.
///
/// A and B fix theirs when they are declared and the accumulator leaves it undefined until an
/// access names one, so the fragment's own answer wins where it has one and the operation's is
/// what is left. Same precedence as the C++ backend's PTX path.
fn access_layout(matrix: &MatrixType, op_layout: MatrixLayout) -> Option<&'static str> {
    layout_name(matrix.layout).or_else(|| layout_name(op_layout))
}

/// The `MatrixType` a matrix operand points at.
///
/// The conversion has already turned the pointer opaque by the time these ops are rewritten,
/// so the fragment's shape is read out of the operand's type history rather than its current
/// type. Same as the AMDGPU lowering does, and for the same reason.
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

/// Element `i` of `vector`.
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

/// `vector` with `value` written into element `i`.
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

/// An undefined value of `ty`, to build a vector into.
fn poison(ctx: &mut Context, rw: &mut DialectConversionRewriter, ty: TypeHandle) -> Value {
    let op = llvm::PoisonOp::new(ctx, ty);
    insert(ctx, rw, &op)
}

/// The fragment vector as the registers an instruction takes, in order.
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

/// The inverse of [`to_registers`]: the registers an instruction gave back, as the fragment
/// vector.
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

/// Calls the intrinsic `name`, which returns one register per field of `regs_ty`, and takes
/// them apart into the individual values.
///
/// The WMMA loads and the multiply both answer with several registers, which LLVM gives as an
/// anonymous struct; nothing downstream wants the struct itself.
fn call_returning_registers(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    name: &str,
    reg_tys: Vec<TypeHandle>,
    args: Vec<Value>,
) -> Vec<Value> {
    let count = reg_tys.len();
    let result_ty: TypeHandle =
        StructType::get_unnamed(ctx, (reg_tys, StructLayout::Unpacked)).into();

    let arg_tys = args.iter().map(|arg| arg.get_type(ctx)).collect();
    let fn_ty = FuncType::get(ctx, result_ty, arg_tys, false);
    let call = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
    let aggregate = insert(ctx, rw, &call);

    (0..count)
        .map(|field| {
            let op = llvm::ExtractValueOp::new(ctx, aggregate, vec![field as u32])
                .expect("a constant index into the returned registers");
            insert(ctx, rw, &op)
        })
        .collect()
}

/// Calls the valueless intrinsic `name`, which is what a WMMA store is.
fn call_void(ctx: &mut Context, rw: &mut DialectConversionRewriter, name: &str, args: Vec<Value>) {
    let arg_tys = args.iter().map(|arg| arg.get_type(ctx)).collect();
    let void_ty = pliron_llvm::types::VoidType::get(ctx).into();
    let fn_ty = FuncType::get(ctx, void_ty, arg_tys, false);
    let call = llvm::CallIntrinsicOp::new(ctx, name.into(), fn_ty, args);
    rw.insert_op(ctx, &call);
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

/// The stride an instruction takes, which is a signless `i32` however the kernel spelled it.
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

/// Names the element type that has no WMMA form.
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

    // Every element the lane holds gets the value, redundant copies included: a fill has no
    // layout to respect, so the duplication A and B carry costs nothing here.
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

    // The intrinsic is overloaded on the pointer, so the mangled name carries its address
    // space. Everything reaching here is generic: the shared-memory lowering casts its slices
    // back to the generic space and `InferAddressSpaces` puts them back afterwards.
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
    // A store is always of a `d` fragment, whose layout the instruction takes; unlike a load
    // there is no declared fragment layout to prefer, since only an accumulator is ever
    // stored and its own is undefined.
    let Some(layout) = layout_name(op_layout).or_else(|| layout_name(ty.layout)) else {
        return input_err!(op.loc(ctx), MatrixLayoutUnknown(ty.ident));
    };

    let frag_ty = fragment_ty(ctx, &ty);
    let elem = cube_type_to_llvm(ctx, ty.elem_ty);
    let reg_ty = register_ty(ctx, frag, elem);
    let stride = stride_as_i32(ctx, rw, stride);

    let value = load_fragment(ctx, rw, matrix, frag_ty);
    let regs = to_registers(ctx, rw, value, frag, reg_ty);

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
    // Both operand layouts are baked into the instruction's name, so a fragment that never
    // declared one cannot be multiplied -- there is no runtime choice to make.
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

    // With `f16` inputs the instruction is identified by its accumulator and result types
    // rather than by its operands, which is why only the C/D name appears -- and twice, since
    // CubeCL gives `c` and `d` one type.
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

    // Two fragments of the same ident hold the same elements in the same places whatever their
    // width -- that is what makes `nvcuda::wmma`'s own accumulator conversion element-wise --
    // so a cast within one ident is a conversion of the vector and nothing more. Across idents
    // the layouts are unrelated and opaque, so there is nothing to convert between.
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
        // Same width, so the two are one LLVM type under two CubeCL names and there is
        // nothing to emit -- `fpext` and `fptrunc` each require a change of width and reject
        // a source and destination of the same type. The one same-width pair that is a real
        // conversion, `f16` to `bf16`, cannot arrive: `bf16` has no WMMA form here and
        // `fragment_of` refused it above.
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

/// A fragment whose layout an instruction needs and which never declared one.
#[derive(Debug, Error)]
#[error(
    "the {0} fragment has no layout, and a WMMA instruction names the layout of what it reads; \
     declare the fragment row- or column-major, or give the access one"
)]
pub struct MatrixLayoutUnknown(MatrixIdent);

/// The letter a fragment goes by in an instruction's name.
fn fragment_name(ident: MatrixIdent) -> &'static str {
    match ident {
        MatrixIdent::A => "a",
        MatrixIdent::B => "b",
        MatrixIdent::Accumulator => "c",
    }
}

/// Lowers the manual `mma.sync` operations, which this backend does not implement.
///
/// The cooperative API above goes through `wmma`, whose fragments are opaque; the manual API is
/// the other family, where a kernel addresses the registers itself against the documented
/// `mma.sync` layouts. The runtime does not advertise it (see `restrict_to_llvm_backend`), so
/// these report rather than emitting something that would compile and be wrong.
macro_rules! unsupported_manual_op {
    ($fn_name:ident, $cube_op:ty) => {
        pub(crate) fn $fn_name(
            op: &$cube_op,
            ctx: &mut Context,
            _rw: &mut DialectConversionRewriter,
            _operands_info: &OperandsInfo,
        ) -> Result<()> {
            input_err!(op.loc(ctx), MatrixManualUnsupported(stringify!($cube_op)))
        }
    };
}

unsupported_manual_op!(row_index, RowIndexOp);
unsupported_manual_op!(col_index, ColIndexOp);
unsupported_manual_op!(mma_manual, MmaManualOp);

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

    /// The register counts are what the backend type-checks a call against, so a wrong one
    /// fails to select rather than computing a wrong answer. These are LLVM's own, from
    /// `IntrinsicsNVVM.td`.
    #[test]
    fn the_fragments_are_the_shapes_the_intrinsics_declare() {
        let ctx = Context::default();
        let f16: TypeHandle = Float16Type::get(&ctx).into();
        let f32: TypeHandle = Float32Type::get(&ctx).into();

        // A and B: eight `<2 x half>`, so sixteen elements a lane.
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

        // The accumulator holds eight elements either way, which is what lets a cast between
        // the two widths be element-wise.
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

    /// An A or B fragment of `f32`, or an accumulator of `f16` in an A slot, has no WMMA form;
    /// the ops refuse it rather than sizing a vector no instruction takes.
    #[test]
    fn a_fragment_no_instruction_takes_is_refused() {
        let ctx = Context::default();
        let f32: TypeHandle = Float32Type::get(&ctx).into();

        assert_eq!(fragment_of(&ctx, &matrix(MatrixIdent::A, f32)), None);
        assert_eq!(fragment_of(&ctx, &matrix(MatrixIdent::B, f32)), None);
    }

    /// A declares its layout, an accumulator leaves it undefined until an access names one.
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
