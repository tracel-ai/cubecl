use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::FunctionType;

use super::prelude::*;
use super::to_llvm::cube_type_to_llvm;

#[op_interface_impl]
impl ToLLVMDialect for FuncOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let signature = self.get_type(ctx);
        let (arg_types, res_types) = {
            let signature = signature.deref(ctx);
            let func_ty = type_cast::<dyn FunctionTypeInterface>(&*signature)
                .expect("FuncOp type must be a function type");
            (func_ty.arg_types(), func_ty.res_types())
        };

        let new_args: Vec<TypeHandle> = arg_types
            .into_iter()
            .map(|ty| cube_type_to_llvm(ctx, ty))
            .collect();
        let new_res: Vec<TypeHandle> = res_types
            .into_iter()
            .map(|ty| cube_type_to_llvm(ctx, ty))
            .collect();

        let new_signature = FunctionType::get(ctx, new_args.clone(), new_res);
        self.set_attr_func_type(ctx, TypeAttr::new(new_signature.into()));

        // Keep the entry-block arguments in sync with the converted signature.
        let entry = self.get_entry_block(ctx);
        let args: Vec<Value> = entry.deref(ctx).arguments().collect();
        for (arg, new_ty) in args.into_iter().zip(new_args) {
            rewriter.set_value_type(ctx, arg, new_ty);
        }
        Ok(())
    }
}
