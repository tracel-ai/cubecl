use cubecl_core::ir::{Scope, prelude::*, verify_op_succ};
use pliron::derive::op_interface;

#[op_interface]
pub trait LowerOp {
    verify_op_succ!();
    fn should_lower(&self, _ctx: &Context) -> bool {
        true
    }
    fn lower(&self, scope: &Scope) -> Vec<Value>;
}

pub type LowerOpsSpirvPass = MatchRewritePass<LowerOpsSpirv>;

#[derive(Default)]
pub struct LowerOpsSpirv;

impl MatchRewrite for LowerOpsSpirv {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_cast::<dyn LowerOp>(&*op.dyn_op(ctx)).is_some_and(|it| it.should_lower(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let dyn_op = op.dyn_op(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lower = op_cast::<dyn LowerOp>(&*dyn_op).unwrap();
        let new_values = lower.lower(&scope);
        rewriter.replace_operation_with_values(ctx, op, new_values);

        Ok(())
    }
}

macro_rules! lower_unop {
    ($ty: ty, $name: ident, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::lower::LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                $crate::lower::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let input = self.get_operand(scope.ctx());
                scope.register_value_type::<T, S>(input);
                vec![$name::expand::<T, S>(scope, input.into()).read_value(scope)]
            }
        }
    };
    ($ty: ty, $name: ident) => {
        lower_unop!($ty, $name, |_, _| true);
    };
}
pub(super) use lower_unop;

macro_rules! lower_binop {
    ($ty: ty, $name: ident, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::lower::LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                $crate::lower::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                use cubecl_core::ir::dialect::OperationPtrExt;
                define_scalar!(T);
                define_size!(S);
                let lhs = self.get_operation().operand(scope.ctx(), 0);
                let rhs = self.get_operation().operand(scope.ctx(), 1);
                scope.register_value_type::<T, S>(lhs);
                vec![$name::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)]
            }
        }
    };
    ($ty: ty, $name: ident) => {
        lower_binop!($ty, $name, |_, _| true);
    };
}
pub(crate) use lower_binop;

pub(crate) fn closure_inference_hack<T, R>(
    val: &T,
    ctx: &Context,
    func: impl FnOnce(&T, &Context) -> R,
) -> R {
    func(val, ctx)
}
