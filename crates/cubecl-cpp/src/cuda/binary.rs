use cubecl_core::ir::{
    dialect::math::{Dp4aOp, SaturatingSAddOp, SaturatingSSubOp},
    interfaces::TypedExt,
    prelude::*,
};
use cubecl_core::{frontend::polyfills::expand_dp4a_polyfill, ir::Scope};

use crate::{
    cuda::{cuda_op_with_out, ptx_with_out},
    shared::{CompilationOptions, lowering::LowerOp},
    target::Cuda,
};

cuda_op_with_out!(Dp4aOp, |op, ctx| {
    let a = op.a(ctx).name(ctx);
    let b = op.b(ctx).name(ctx);
    let c = op.c(ctx).name(ctx);
    format!("__dp4a({a}, {b}, {c})")
});

#[op_interface_impl]
impl LowerOp<Cuda> for Dp4aOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        !ctx.aux_ty::<CompilationOptions>().supports_features.dp4a
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        vec![expand_dp4a_polyfill(
            scope,
            self.a(ctx),
            self.b(ctx),
            self.c(ctx),
        )]
    }
}

ptx_with_out!(
    SaturatingSAddOp,
    |_, _| "add.sat.s32 $0, $1, $2;".into(),
    |op, ctx| op.result_type(ctx).is_int_of_width(ctx, 32)
        && op.result_type(ctx).is_signed_int(ctx)
);
ptx_with_out!(
    SaturatingSSubOp,
    |_, _| "sub.sat.s32 $0, $1, $2;".into(),
    |op, ctx| op.result_type(ctx).is_int_of_width(ctx, 32)
        && op.result_type(ctx).is_signed_int(ctx)
);
