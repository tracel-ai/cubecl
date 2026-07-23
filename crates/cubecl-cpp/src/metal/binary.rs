use cubecl_core::{
    frontend::polyfills::*,
    ir::{
        dialect::{cmp::*, math::*},
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};

use crate::{metal::metal_op_with_out, shared::lowering::LowerOp, target::Metal};

metal_op_with_out!(SaturatingSAddOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("addsat({lhs}, {rhs})")
});
metal_op_with_out!(SaturatingUAddOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("addsat({lhs}, {rhs})")
});

metal_op_with_out!(SaturatingSSubOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("subsat({lhs}, {rhs})")
});
metal_op_with_out!(SaturatingUSubOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("subsat({lhs}, {rhs})")
});

metal_op_with_out!(SMinOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("min({lhs}, {rhs})")
});
metal_op_with_out!(UMinOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("min({lhs}, {rhs})")
});
metal_op_with_out!(FMinOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("min({lhs}, {rhs})")
});

metal_op_with_out!(SMaxOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("max({lhs}, {rhs})")
});
metal_op_with_out!(UMaxOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("max({lhs}, {rhs})")
});
metal_op_with_out!(FMaxOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("max({lhs}, {rhs})")
});

metal_op_with_out!(PowfOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("pow({lhs}, {rhs})")
});

metal_op_with_out!(PowiOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("pow({lhs}, {rhs})")
});

metal_op_with_out!(HypotOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("length(float2({lhs}, {rhs}))")
});

metal_op_with_out!(RhypotOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("rsqrt({lhs} * {lhs} + {rhs} * {rhs}))")
});

#[op_interface_impl]
impl LowerOp<Metal> for SMulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let lhs = self.lhs(ctx);
        let val = if lhs.size_bits(ctx) == 32 {
            expand_s_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}

#[op_interface_impl]
impl LowerOp<Metal> for UMulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let lhs = self.lhs(ctx);
        let val = if lhs.size_bits(ctx) == 32 {
            expand_u_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}
