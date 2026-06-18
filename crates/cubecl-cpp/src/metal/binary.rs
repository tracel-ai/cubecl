use cubecl_core::{
    ir::{
        dialect::{
            cmp::{MaxOp, MinOp},
            math::{HypotOp, MulHiOp, PowfOp, PowiOp, RhypotOp, SaturatingAddOp, SaturatingSubOp},
        },
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};

use crate::{metal::metal_op_with_out, shared::lowering::LowerOp, target::Metal};

metal_op_with_out!(SaturatingAddOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("addsat({lhs}, {rhs})")
});

metal_op_with_out!(SaturatingSubOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("subsat({lhs}, {rhs})")
});

metal_op_with_out!(MinOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("min({lhs}, {rhs})")
});

metal_op_with_out!(MaxOp, |op, ctx| {
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
impl LowerOp<Metal> for MulHiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let lhs = self.lhs(ctx);
        let val = if lhs.is_int_of_width(ctx, 32) || lhs.is_uint_of_width(ctx, 64) {
            expand_himul_64(scope, lhs, self.rhs(ctx))
        } else {
            expand_himul_sim(scope, lhs, self.rhs(ctx))
        };
        vec![val]
    }
}
