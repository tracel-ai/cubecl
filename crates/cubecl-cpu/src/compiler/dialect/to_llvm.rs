pub use super::prelude::*;
use cubecl_core::ir::types::PointerType as CubePointerType;
use cubecl_core::ir::types::scalar::{BoolType, IndexType};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron_llvm::types::PointerType as LlvmPointerType;

/// LLVM width of a `cube.index`. `IndexType` is `size_of::<u64>()`, so it maps to `i64`.
const INDEX_WIDTH: u32 = 64;

/// Which LLVM type a cubecl type maps to.
enum LlvmTypeKind {
    /// A signless integer of the given width (LLVM integers carry no signedness).
    SignlessInt(u32),
    /// An opaque LLVM pointer
    Pointer,
    /// No conversion applies; keep the type as-is.
    Identity,
}

/// Convert a cubecl type to its LLVM-dialect equivalent, or return it unchanged when no
/// conversion applies.
pub fn cube_type_to_llvm(ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
    let kind = {
        let ty = ty.deref(ctx);
        if let Some(int) = ty.downcast_ref::<IntegerType>() {
            if int.signedness() != Signedness::Signless {
                LlvmTypeKind::SignlessInt(int.width())
            } else {
                LlvmTypeKind::Identity
            }
        } else if ty.is::<BoolType>() {
            LlvmTypeKind::SignlessInt(1)
        } else if ty.is::<IndexType>() {
            LlvmTypeKind::SignlessInt(INDEX_WIDTH)
        } else if ty.is::<CubePointerType>() {
            LlvmTypeKind::Pointer
        } else {
            LlvmTypeKind::Identity
        }
    };
    match kind {
        LlvmTypeKind::SignlessInt(width) => {
            IntegerType::get(ctx, width, Signedness::Signless).into()
        }
        LlvmTypeKind::Pointer => LlvmPointerType::get(ctx, 0).into(),
        LlvmTypeKind::Identity => ty,
    }
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
        let ty = ty.deref(ctx);
        ty.is::<CubePointerType>()
            || ty.is::<BoolType>()
            || ty.is::<IndexType>()
            || matches!(
                ty.downcast_ref::<IntegerType>(),
                Some(int) if int.signedness() != Signedness::Signless
            )
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        Ok(cube_type_to_llvm(ctx, ty))
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
