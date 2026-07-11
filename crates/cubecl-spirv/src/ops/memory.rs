use cubecl_ir::{
    AddressSpace,
    dialect::memory::{self, DeclareVariableOp, IndexOp},
    interfaces::TypedExt,
    prelude::*,
};
use pliron::builtin::ops::FuncOp;
use pliron_spirv::ops::{self, InBoundsAccessChainOp, VariableOp};
use rspirv::spirv::{MemoryAccess, StorageClass};

use crate::{ops::to_spirv_dialect::ToSpirvDialectOp, types::ty_to_spirv_dialect};

#[op_interface_impl]
impl ToSpirvDialectOp for DeclareVariableOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        assert_eq!(self.addr_space(ctx).0, AddressSpace::Local, "TODO");

        let op = self.get_operation();
        let func = find_parent_func(ctx, op);
        rewriter.set_insertion_point_to_block_start(func.get_entry_block(ctx));

        let result_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let var = VariableOp::new(ctx, result_ty, StorageClass::Function, None);
        rewriter.insert_op(ctx, &var);
        rewriter.replace_operation(ctx, op, var.get_operation());

        Ok(())
    }
}

fn find_parent_func(ctx: &Context, op: Ptr<Operation>) -> FuncOp {
    let mut op = op;
    while !op.is_op::<FuncOp>(ctx) {
        let parent = op.deref(ctx).get_parent_op(ctx);
        op = parent.expect("Should have parent");
    }
    op.as_op::<FuncOp>(ctx).unwrap()
}

#[op_interface_impl]
impl ToSpirvDialectOp for IndexOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let base = self.base(ctx);
        let index = self.index(ctx);
        let result_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let access_chain = InBoundsAccessChainOp::new(ctx, result_ty, base, vec![index]);
        rewriter.insert_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for memory::LoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let align = self.get_result(ctx).align(ctx) as u32;
        let ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let access_chain = ops::LoadOp::new(ctx, ty, ptr, MemoryAccess::ALIGNED, Some(align));
        rewriter.insert_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for memory::StoreOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ptr = self.ptr(ctx);
        let value = self.value(ctx);
        let align = value.align(ctx) as u32;
        let access_chain = ops::StoreOp::new(ctx, ptr, value, MemoryAccess::ALIGNED, Some(align));
        rewriter.insert_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for memory::CopyOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let src = self.source(ctx);
        let dest = self.destination(ctx);
        let align = src.unwrap_ptr(ctx).align(ctx) as u32;
        let access_chain = ops::CopyMemoryOp::new(
            ctx,
            dest,
            src,
            MemoryAccess::ALIGNED,
            Some(align),
            MemoryAccess::NONE,
            None,
        );
        rewriter.insert_op(ctx, &access_chain);
        rewriter.replace_operation(ctx, self.get_operation(), access_chain.get_operation());

        Ok(())
    }
}
