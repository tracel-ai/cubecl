use cubecl_ir::dialect::cmp::*;

use crate::compiler::wgsl::to_wgsl::wgsl_op_with_out;

wgsl_op_with_out!(SMinOp, |op, ctx| {
    format!("min({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(UMinOp, |op, ctx| {
    format!("min({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FMinOp, |op, ctx| {
    format!("min({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SMaxOp, |op, ctx| {
    format!("max({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(UMaxOp, |op, ctx| {
    format!("max({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FMaxOp, |op, ctx| {
    format!("max({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SClampOp, |op, ctx| {
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({}, {min}, {max})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(UClampOp, |op, ctx| {
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({}, {min}, {max})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(FClampOp, |op, ctx| {
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("clamp({}, {min}, {max})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SLessThanOp, |op, ctx| {
    format!("{} < {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(ULessThanOp, |op, ctx| {
    format!("{} < {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FLessThanOp, |op, ctx| {
    format!("{} < {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SLessThanOrEqualOp, |op, ctx| {
    format!("{} <= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(ULessThanOrEqualOp, |op, ctx| {
    format!("{} <= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FLessThanOrEqualOp, |op, ctx| {
    format!("{} <= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SGreaterThanOp, |op, ctx| {
    format!("{} > {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(UGreaterThanOp, |op, ctx| {
    format!("{} > {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FGreaterThanOp, |op, ctx| {
    format!("{} > {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(SGreaterThanOrEqualOp, |op, ctx| {
    format!("{} >= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(UGreaterThanOrEqualOp, |op, ctx| {
    format!("{} >= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FGreaterThanOrEqualOp, |op, ctx| {
    format!("{} >= {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(IEqualOp, |op, ctx| {
    format!("{} == {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FEqualOp, |op, ctx| {
    format!("{} == {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(BoolEqualOp, |op, ctx| {
    format!("{} == {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(INotEqualOp, |op, ctx| {
    format!("{} != {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FNotEqualOp, |op, ctx| {
    format!("{} != {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(BoolNotEqualOp, |op, ctx| {
    format!("{} != {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
