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
    // An intrinsic returning one value returns it bare; only several become a struct. Wrapping
    // a single one anyway would name a signature LLVM does not have -- `ldmatrix.x1` is the
    // one that reaches this -- so the two cases are built differently.
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

    // The intrinsic is overloaded on the pointer, and both the mangled name and the qualifier
    // on the instruction it selects come from that pointer's address space. See
    // [`origin_address_space`] for why the generic pointer the rest of the pipeline carries is
    // narrowed here rather than left to `InferAddressSpaces`.
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

    // Narrowed for the same reason as the load's source.
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

/// The manual `mma.sync` API.
///
/// The other matrix family, and the opposite of the cooperative one above in every way that
/// matters here: the fragment layout is documented rather than opaque, a kernel addresses its
/// own registers against it (see the `row_index`/`col_index` polyfill in
/// [`shared::matrix`](crate::shared::matrix)), and the shapes are the narrow `m16n8k*` ones
/// the tensor cores actually execute rather than the wide ones `wmma` composes out of them.
///
/// So there is nothing to move here: the registers arrive as the array the frontend built and
/// go straight into the instruction.
///
/// How a fragment's registers are read out of the vector holding them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegisterForm {
    /// One register is `n` elements of the fragment's own type, which is how the 16 bit floats
    /// are passed: `<2 x half>`.
    Packed(usize),
    /// One register is one element, which is how a 32 bit accumulator is passed.
    Scalar,
    /// One register is an opaque `i32` the fragment is reinterpreted into, which is how the
    /// narrow integers are passed -- four `i8` to a register, with no vector type for them.
    Word,
}

/// The PTX type name `elem` goes by in an `mma.sync`, and how its registers are read.
fn mma_type(ctx: &Context, elem: TypeHandle) -> Option<(&'static str, RegisterForm)> {
    if elem.is_float16(ctx) {
        Some(("f16", RegisterForm::Packed(2)))
    } else if elem.is_float32(ctx) {
        Some(("f32", RegisterForm::Scalar))
    } else if elem.is_int(ctx) && elem.size_bits(ctx) == 8 {
        // Whether the eight bits are read as signed or unsigned is part of the instruction's
        // name, so the two are different `mma.sync`s over the same registers.
        let name = if elem.is_signed_int(ctx) { "s8" } else { "u8" };
        Some((name, RegisterForm::Word))
    } else if elem.is_int(ctx) && elem.size_bits(ctx) == 32 {
        Some(("s32", RegisterForm::Scalar))
    } else {
        None
    }
}

/// A signless `i32`, which is what an opaque register is.
fn word_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

/// The registers of `vector`, as `form` says to read them.
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
            // No vector type covers four `i8` in a register, so the whole fragment is
            // reinterpreted into words and read element-wise.
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

/// The inverse of [`registers_of`]: `regs` gathered back into a vector of `vector_ty`.
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

/// The part of an `mma.sync` intrinsic's name that says which types it multiplies.
///
/// LLVM identifies these ops by whichever fragments actually distinguish them, which is not
/// always the operands: with `f16` inputs the accumulator and result do it, since the same
/// instruction takes either accumulator width. Getting this wrong names an intrinsic that does
/// not exist, so it follows `MMA_SIGNATURE` in `IntrinsicsNVVM.td` exactly.
fn mma_signature(a: &str, b: &str, cd: &str) -> String {
    if a == "f16" {
        // Identified by the accumulator and the result, which CubeCL gives one type.
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

    // A and B are always row- and column-major here: that is the one layout `mma.sync` takes
    // for these shapes, and what `TargetProperties::mma` tells a kernel to arrange its
    // registers for.
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

/// The address space `ldmatrix` and `stmatrix` read and write.
///
/// Both are shared-memory instructions -- `.shared::cta` in PTX -- so the generic pointer the
/// rest of the pipeline carries is cast down to it, which is what the C++ backend's
/// `generic_to_shared` does before its inline asm.
const SHARED_ADDRESS_SPACE: u32 = 3;

/// The address space a buffer lives in, i.e. what a kernel argument is retyped to in
/// [`abi`](super::abi).
const GLOBAL_ADDRESS_SPACE: u32 = 1;

/// The generic space, which is where the rest of the pipeline carries every pointer.
const GENERIC_ADDRESS_SPACE: u32 = 0;

/// The address space of `value`, when it is a pointer at all.
fn address_space(ctx: &Context, value: Value) -> Option<u32> {
    value
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<LlvmPointerType>()
        .map(LlvmPointerType::address_space)
}

/// The address space `ptr` provably points into, looked through the generic space.
///
/// A `wmma` load or store picks its `.shared` / `.global` qualifier from the address space of
/// the pointer it is *given* -- see `AS_match` in LLVM's `NVPTXIntrinsics.td` -- and nothing
/// puts a generic one back: `InferAddressSpaces` only rewrites intrinsics the target lists in
/// `collectFlatAddressOperands`, which for NVPTX is `isspacep` and `prefetch.tensormap` and
/// not these. So a generic pointer here costs the qualifier, and the qualifier is most of the
/// instruction. On one cmma GEMM the eighty generic fragment loads came out of `ptxas` as 320
/// scalar `LD.E` and 256 `MOVM` transposes assembling the tiles by hand, against 64 `LDSM`
/// once the pointers were shared.
///
/// Recovering it is a walk back to whatever the address was derived from, through the
/// `getelementptr`s that offset into a tile and the `addrspacecast` that
/// [`SliceSharedOp`](crate::shared::shared_memory) inserts, stopping at the first pointer that
/// is already in a real address space. `None` for a pointer whose origin is not one of those,
/// which is the answer that leaves the call generic and correct.
fn origin_address_space(ctx: &Context, ptr: Value) -> Option<u32> {
    let mut ptr = ptr;
    // Bounded by the length of the chain, which is acyclic: every step is to an operand of the
    // op defining the current value.
    loop {
        match address_space(ctx, ptr) {
            // Generic says nothing about where it came from, so keep walking.
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
        // Operand 0 of both: the base of the `getelementptr`, the argument of the cast.
        ptr = op.deref(ctx).get_operand(0);
    }
}

/// `ptr` in the address space it provably points into, when that is one a `wmma` load or store
/// has a qualifier for.
///
/// The cast is a single `cvta.to.shared` / `cvta.to.global`, and usually not even that: it
/// undoes an `addrspacecast` the pointer already went through, which LLVM folds. What it buys
/// is the qualifier on the instruction it feeds.
///
/// Shared is the space that arises in practice: a buffer is a kernel argument, and
/// [`PtxKernelParams`](super::abi) retypes those to the global space later, in the LLVM module
/// rather than here, so a fragment read straight out of one still looks generic at this point
/// and stays generic. The arm is here because the answer is the same either way.
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

/// `ptr` in the shared address space, which is the only one these instructions take.
fn as_shared(ctx: &mut Context, rw: &mut DialectConversionRewriter, ptr: Value) -> Value {
    let shared_ty: TypeHandle = LlvmPointerType::get(ctx, SHARED_ADDRESS_SPACE).into();
    if ptr.get_type(ctx) == shared_ty {
        return ptr;
    }
    let op = llvm::AddrSpaceCastOp::new(ctx, ptr, shared_ty);
    insert(ctx, rw, &op)
}

/// The `.trans` qualifier, which swaps the rows and columns of each 8x8 tile as it moves.
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

    // One instruction moves `factor` 8x8 tiles of 16 bit elements, a register per tile per
    // lane, whatever the elements stand for -- `b16` is a width, not a type.
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

/// `row_index` and `col_index` are answered before this conversion runs, by the polyfill in
/// [`shared::matrix`](crate::shared::matrix) that expands `CubeCL`'s own formulas. Reaching here
/// means that pass did not run, which is a bug in the pipeline rather than in the kernel.
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
