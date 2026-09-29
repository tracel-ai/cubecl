use cubecl_ir::{Scope, dialect::cmp::*, prelude::*};

use crate::compiler::wgsl::to_wgsl::wgsl_op_with_out;

wgsl_op_with_out!(SMinOp, UMinOp, FMinOp; |op, ctx| {
    format!("min({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SMaxOp, UMaxOp, FMaxOp; |op, ctx| {
    format!("max({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SClampOp, UClampOp, FClampOp; |op, ctx| {
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({}, {min}, {max})", op.input(ctx).name(ctx))
});

#[op_interface_impl]
impl crate::compiler::wgsl::lower::LowerOp for FClampNanOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let value = cubecl_core::frontend::polyfills::expand_clamp_nan(
            scope,
            self.input(ctx),
            self.min(ctx),
            self.max(ctx),
        );
        vec![value]
    }
}

wgsl_op_with_out!(SLessThanOp, ULessThanOp, FLessThanOp; |op, ctx| {
    format!("{} < {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(SLessThanOrEqualOp, ULessThanOrEqualOp, FLessThanOrEqualOp; |op, ctx| {
    format!("{} <= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(SGreaterThanOp, UGreaterThanOp, FGreaterThanOp; |op, ctx| {
    format!("{} > {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(SGreaterThanOrEqualOp, UGreaterThanOrEqualOp, FGreaterThanOrEqualOp; |op, ctx| {
    format!("{} >= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(IEqualOp, FEqualOp, BoolEqualOp; |op, ctx| {
    format!("{} == {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(INotEqualOp, FNotEqualOp, BoolNotEqualOp; |op, ctx| {
    format!("{} != {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

#[op_interface_impl]
impl crate::compiler::wgsl::lower::LowerOp for FMinNanOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let value =
            cubecl_core::frontend::polyfills::expand_min_nan(scope, self.lhs(ctx), self.rhs(ctx));
        vec![value]
    }
}

#[op_interface_impl]
impl crate::compiler::wgsl::lower::LowerOp for FMaxNanOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let value =
            cubecl_core::frontend::polyfills::expand_max_nan(scope, self.lhs(ctx), self.rhs(ctx));
        vec![value]
    }
}
