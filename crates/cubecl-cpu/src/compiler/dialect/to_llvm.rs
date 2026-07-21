pub use super::prelude::*;

#[type_interface]
pub trait CubeToLLVMType {
    verify_ty_succ!();
    /// Build the LLVM-dialect type equivalent to this cube type.
    fn convert(&self, ctx: &Context) -> TypeHandle;
}

/// Convert a cubecl type to its LLVM-dialect equivalent, or return it unchanged when no
/// conversion applies.
pub fn cube_type_to_llvm(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    type_cast::<dyn CubeToLLVMType>(&*ty.deref(ctx))
        .map(|convertible| convertible.convert(ctx))
        .unwrap_or(ty)
}

#[op_interface]
pub trait ToLLVMDialect {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

pub type CubeToLLVMPass = DialectConversionPass<CubeToLLVM>;

#[derive(Default)]
pub struct CubeToLLVM;

impl DialectConversion for CubeToLLVM {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToLLVMDialect>(&*Operation::get_op_dyn(op, ctx))
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        type_cast::<dyn CubeToLLVMType>(&*ty.deref(ctx)).is_some()
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let ty = ty.deref(ctx);
        let ty = type_cast::<dyn CubeToLLVMType>(&*ty).unwrap();
        let ty = ty.convert(ctx);
        Ok(ty)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_llvm_op = op_cast::<dyn ToLLVMDialect>(&*op_dyn)
            .expect("Matched Op must implement ToLLVMDialect");
        to_llvm_op.rewrite(ctx, rewriter, operands_info)
    }
}
