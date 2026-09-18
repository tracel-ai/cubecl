use crate::prelude::*;
use cubecl_core::ir::dialect::{
    barrier::{
        ArriveAndExpectTxOp, ArriveAndWaitOp, ArriveOp, CommitCopyAsyncOp, ExpectTxOp, InitOp,
        WaitOp, WaitParityOp,
    },
    general::{CastOp, CommentOp, CopyOp, FreeOp, PrintfOp, ReinterpretCastOp, SelectOp},
};
use pliron_llvm::{
    attributes::BytesAttr,
    function_call_utils::lookup_or_insert_function,
    ops::{FPExtOp, FPTruncOp},
    types::ArrayType,
};
use std::hash::{DefaultHasher, Hash, Hasher};

fn int_repr(ctx: &Context, ty: TypeHandle) -> Option<(u32, bool)> {
    let ty = ty.deref(ctx);
    if let Some(int) = ty.downcast_ref::<IntegerType>() {
        Some((int.width(), int.signedness() == Signedness::Signed))
    } else if ty.is::<BoolType>() {
        Some((1, false))
    } else if ty.is::<IndexType>() {
        Some((index_width(ctx), false))
    } else {
        None
    }
}

fn cast_int_to_int(
    cast_op: &CastOp,
    is_signed: bool,
    in_width: u32,
    out_width: u32,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) -> Result<()> {
    let out_ty = cast_op.get_result(ctx).get_type(ctx);
    let input = cast_op.input(ctx);
    let old_op = cast_op.get_operation();

    let out_ty = cube_type_to_llvm(ctx, out_ty);

    // Any nonzero integer converts to true.
    if out_width == 1 && in_width > 1 {
        let zero = insert_zero(ctx, rewriter, input);
        let cmp = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::NE, input, zero);
        rewriter.insert_op(ctx, &cmp);
        rewriter.replace_operation_with_values(ctx, old_op, vec![cmp.get_result(ctx)]);
    } else if in_width == out_width {
        rewriter.replace_operation_with_values(ctx, old_op, vec![input]);
    } else if out_width > in_width {
        let op: &dyn OneResultInterface = if is_signed {
            &llvm::SExtOp::new(ctx, input, out_ty)
        } else {
            &llvm::ZExtOp::new_with_nneg(ctx, input, out_ty, false)
        };
        rewriter.insert_op(ctx, op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    } else {
        let op = llvm::TruncOp::new(ctx, input, out_ty);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    }

    Ok(())
}

fn insert_zero(ctx: &mut Context, rewriter: &mut DialectConversionRewriter, value: Value) -> Value {
    let zero = constant_op(ctx, ZeroAttr::new(value.get_type(ctx)).into());
    rewriter.insert_op(ctx, &*zero.dyn_op(ctx));
    zero.deref(ctx).get_result(0)
}

fn cast_float_to_int(
    cast_op: &CastOp,
    is_signed: bool,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) {
    let res_ty = cube_type_to_llvm(ctx, cast_op.result_type(ctx));
    let input = cast_op.input(ctx);
    let old_op = cast_op.get_operation();

    // Any nonzero float, including NaN, converts to true.
    if cast_op.result_type(ctx).scalar_ty(ctx).is_bool(ctx) {
        let zero = insert_zero(ctx, rewriter, input);
        let cmp = llvm::FCmpOp::new(ctx, FCmpPredicateAttr::UNE, input, zero);
        cmp.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        rewriter.insert_op(ctx, &cmp);
        rewriter.replace_operation_with_values(ctx, old_op, vec![cmp.get_result(ctx)]);
        return;
    }

    let op: &dyn OneResultInterface = if is_signed {
        &llvm::FPToSIOp::new(ctx, input, res_ty)
    } else {
        &llvm::FPToUIOp::new(ctx, input, res_ty)
    };
    rewriter.insert_op(ctx, op);
    rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
}

fn cast_int_to_float(
    cast_op: &CastOp,
    is_signed: bool,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) {
    let res_ty = cube_type_to_llvm(ctx, cast_op.result_type(ctx));
    let input = cast_op.input(ctx);
    let old_op = cast_op.get_operation();

    let op: &dyn OneResultInterface = if is_signed {
        &llvm::SIToFPOp::new(ctx, input, res_ty)
    } else {
        &llvm::UIToFPOp::new_with_nneg(ctx, input, res_ty, false)
    };
    rewriter.insert_op(ctx, op);
    rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
}

fn cast_float_to_float(
    cast_op: &CastOp,
    in_ty: TypeHandle,
    out_ty: TypeHandle,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) {
    let res_ty = cube_type_to_llvm(ctx, cast_op.result_type(ctx));
    let input = cast_op.input(ctx);
    let old_op = cast_op.get_operation();
    let input_size = in_ty.size(ctx);
    let output_size = out_ty.size(ctx);

    if input_size > output_size {
        let op = FPTruncOp::new(ctx, input, res_ty);
        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    } else if input_size < output_size {
        let op = FPExtOp::new(ctx, input, res_ty);
        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, old_op, vec![op.get_result(ctx)]);
    } else {
        rewriter.replace_operation_with_values(ctx, old_op, vec![input]);
    }
}

fn extract_elem_type(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    if let Some(ty) = ty.deref(ctx).downcast_ref::<LlvmVectorType>() {
        ty.elem_type()
    } else if let Some(ty) = ty.deref(ctx).downcast_ref::<CubeVectorType>() {
        ty.scalar_type(ctx)
    } else {
        ty
    }
}

#[op_interface_impl]
impl ToLLVMDialect for CastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let in_ty = operands_info
            .lookup_most_recent_type(self.input(ctx))
            .unwrap_or(self.input(ctx).get_type(ctx));
        let in_ty = extract_elem_type(ctx, in_ty);
        let out_ty = self.get_result(ctx).get_type(ctx);
        let out_ty = extract_elem_type(ctx, out_ty);

        if let (Some((in_width, in_signed)), Some((out_width, _))) =
            (int_repr(ctx, in_ty), int_repr(ctx, out_ty))
        {
            return cast_int_to_int(self, in_signed, in_width, out_width, ctx, rewriter);
        }

        if in_ty.is_float(ctx)
            && let Some((_, out_signed)) = int_repr(ctx, out_ty)
        {
            cast_float_to_int(self, out_signed, ctx, rewriter);
        }

        if let Some((_, out_signed)) = int_repr(ctx, in_ty)
            && out_ty.is_float(ctx)
        {
            cast_int_to_float(self, out_signed, ctx, rewriter);
        }

        if in_ty.is_float(ctx) && out_ty.is_float(ctx) {
            cast_float_to_float(self, in_ty, out_ty, ctx, rewriter);
        }

        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for CopyOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.value(ctx);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![value]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for ReinterpretCastOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let out_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));
        let old_op = self.get_operation();

        if out_ty.deref(ctx).is::<LlvmPointerType>() {
            rewriter.replace_operation_with_values(ctx, old_op, vec![input]);
            return Ok(());
        }

        let bitcast = llvm::BitcastOp::new(ctx, input, out_ty);
        rewriter.insert_op(ctx, &bitcast);
        rewriter.replace_operation_with_values(ctx, old_op, vec![bitcast.get_result(ctx)]);
        Ok(())
    }
}

#[op_interface_impl]
impl ToLLVMDialect for SelectOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let new_op = llvm::SelectOp::new(
            ctx,
            self.condition(ctx),
            self.true_value(ctx),
            self.false_value(ctx),
        );
        rewriter.insert_op(ctx, &new_op);
        rewriter.replace_operation_with_values(
            ctx,
            self.get_operation(),
            vec![new_op.get_result(ctx)],
        );
        Ok(())
    }
}

const PRINTF: &str = "printf";

fn parent_module(ctx: &Context, op: Ptr<Operation>) -> Option<ModuleOp> {
    let mut current = Some(op);
    while let Some(op) = current {
        if let Some(module) = Operation::get_op::<ModuleOp>(op, ctx) {
            return Some(module);
        }
        current = op.deref(ctx).get_parent_op(ctx);
    }
    None
}

fn lookup_or_insert_format_string(
    ctx: &mut Context,
    symbol_tables: &mut SymbolTableCollection,
    module: ModuleOp,
    format_string: &str,
) -> Result<Identifier> {
    let mut hasher = DefaultHasher::new();
    format_string.hash(&mut hasher);
    let name: Identifier = format!("cube_printf_fmt_{:016x}", hasher.finish())
        .try_into()
        .expect("Generated name is a valid identifier");

    let symbol_table = symbol_tables.get_symbol_table(ctx, Box::new(module));
    if symbol_table.lookup(&name).is_some() {
        return Ok(name);
    }

    let mut bytes = format_string.as_bytes().to_vec();
    bytes.push(0);

    let byte_ty = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let array_ty = ArrayType::get(ctx, byte_ty, bytes.len() as u64).into();
    let global = llvm::GlobalOp::new(ctx, name.clone(), array_ty);
    global.set_initializer_value(ctx, BytesAttr::new(bytes).into());
    global.set_attr_llvm_global_linkage(ctx, LinkageAttr::PrivateLinkage);
    symbol_tables
        .get_symbol_table(ctx, Box::new(module))
        .insert(ctx, Box::new(global), None)?;

    Ok(name)
}

/// Variadic C arguments require promotion to at least int or double.
fn promote_vararg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    ty: TypeHandle,
) -> Value {
    if ty.is_float(ctx) {
        if ty.is_float64(ctx) {
            return value;
        }
        let op = llvm::FPExtOp::new(ctx, value, FP64Type::get(ctx).into());
        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        rewriter.insert_op(ctx, &op);
        return op.get_result(ctx);
    }

    if let Some((width, is_signed)) = int_repr(ctx, ty)
        && width < I32_WIDTH
    {
        let i32_ty = IntegerType::get(ctx, I32_WIDTH, Signedness::Signless).into();
        let op: &dyn OneResultInterface = if is_signed {
            &llvm::SExtOp::new(ctx, value, i32_ty)
        } else {
            &llvm::ZExtOp::new_with_nneg(ctx, value, i32_ty, false)
        };
        rewriter.insert_op(ctx, op);
        return op.get_result(ctx);
    }

    value
}

#[op_interface_impl]
impl ToLLVMDialect for PrintfOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let old_op = self.get_operation();
        let module = parent_module(ctx, old_op).expect("Printf op should be inside a module");
        let mut symbol_tables = SymbolTableCollection::new();

        let format_string = self.format_string(ctx).as_str().to_string();
        let global_name =
            lookup_or_insert_format_string(ctx, &mut symbol_tables, module, &format_string)?;

        let ptr_ty = LlvmPointerType::get(ctx, 0).into();
        let i32_ty = IntegerType::get(ctx, I32_WIDTH, Signedness::Signless).into();
        let printf = lookup_or_insert_function(
            ctx,
            &mut symbol_tables,
            Box::new(module),
            PRINTF.try_into().expect("`printf` is a valid identifier"),
            i32_ty,
            vec![ptr_ty],
            true,
        )?;

        let format_ptr = llvm::AddressOfOp::new(ctx, global_name, 0);
        rewriter.insert_op(ctx, &format_ptr);

        let mut args = vec![format_ptr.get_result(ctx)];
        for arg in self.args(ctx) {
            let ty = operands_info
                .lookup_most_recent_type(arg)
                .unwrap_or(arg.get_type(ctx));
            args.push(promote_vararg(ctx, rewriter, arg, ty));
        }

        let call = llvm::CallOp::new(
            ctx,
            CallOpCallable::Direct(printf.get_symbol_name(ctx)),
            printf.get_type(ctx),
            args,
        );
        rewriter.insert_op(ctx, &call);
        rewriter.erase_operation(ctx, old_op);

        Ok(())
    }
}

macro_rules! erase_op {
    ($cube_op:ty) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                rewriter.erase_operation(ctx, self.get_operation());
                Ok(())
            }
        }
    };
}

erase_op!(CommentOp);
erase_op!(FreeOp);
erase_op!(InitOp);
erase_op!(ArriveOp);
erase_op!(ArriveAndExpectTxOp);
erase_op!(CommitCopyAsyncOp);
erase_op!(ExpectTxOp);
erase_op!(WaitOp);
erase_op!(WaitParityOp);
erase_op!(ArriveAndWaitOp);
