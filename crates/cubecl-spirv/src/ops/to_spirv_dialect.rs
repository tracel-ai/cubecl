use cubecl_ir::{
    prelude::{
        Context, DialectConversion, DialectConversionRewriter, OperandsInfo, Operation,
        OperationPtrExt, Ptr, Result, Rewriter,
    },
    rewrite::DialectConversionPass,
    verify_op_succ,
};
use pliron::{
    builtin::ops::{ConstantOp, FuncOp},
    derive::{op_interface, op_interface_impl},
    irbuild::inserter::Inserter,
    op::{Op, op_cast},
    r#type::{TypeHandle, Typed},
};

use crate::{attributes::attr_to_spirv_dialect, types::ty_to_spirv_dialect};

#[op_interface]
pub trait ToSpirvDialectOp {
    verify_op_succ!();
    fn should_convert(&self, _ctx: &Context) -> bool {
        true
    }
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;
}

pub type ToSpirvDialectPass = DialectConversionPass<ToSpirvDialect>;

#[derive(Default)]
pub struct ToSpirvDialect;

impl DialectConversion for ToSpirvDialect {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        let dyn_op = op.dyn_op(ctx);
        op_cast::<dyn ToSpirvDialectOp>(&*dyn_op).is_some_and(|op| op.should_convert(ctx))
    }

    fn can_convert_type(&self, ctx: &Context, handle: TypeHandle) -> bool {
        ty_to_spirv_dialect(ctx, handle) != handle
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let to_spirv_dialect = op_cast::<dyn ToSpirvDialectOp>(&*dyn_op).unwrap();
        to_spirv_dialect.to_spirv_dialect(ctx, rewriter, operands_info)
    }

    fn convert_type(&mut self, ctx: &mut Context, handle: TypeHandle) -> Result<TypeHandle> {
        Ok(ty_to_spirv_dialect(ctx, handle))
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for FuncOp {
    fn should_convert(&self, ctx: &Context) -> bool {
        let ty = self.get_attr_func_type(ctx).unwrap().get_type(ctx);
        ty_to_spirv_dialect(ctx, ty) != ty
    }

    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        _rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let func_ty = self.get_attr_func_type(ctx).unwrap().get_type(ctx);
        let func_ty = ty_to_spirv_dialect(ctx, func_ty);
        self.set_attr_func_type(ctx, func_ty.into());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for ConstantOp {
    fn should_convert(&self, ctx: &Context) -> bool {
        attr_to_spirv_dialect(ctx, &self.get_value(ctx)) != self.get_value(ctx)
    }

    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let attr = attr_to_spirv_dialect(ctx, &self.get_value(ctx));
        let new_const = ConstantOp::new(ctx, attr);
        rewriter.insert_op(ctx, &new_const);
        rewriter.replace_operation(ctx, self.get_operation(), new_const.get_operation());
        Ok(())
    }
}
