use cubecl_core::{
    frontend::polyfills::*,
    ir::{
        dialect::{cmp::*, math::*},
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};

use crate::{
    metal::metal_op_with_out,
    shared::{CppValue, convert::no_msl_bfloat, lowering::LowerOp, unroll::unrolling},
    target::Metal,
};

/// An operand of a signed saturating builtin, spelled in the type MSL overloads it on.
///
/// `metal_stdlib` defines `int8_t` as `signed char`, but `addsat` and `subsat` are overloaded on
/// `char`: an `int8_t` argument promotes the call to `int`, and the result wraps when it is
/// stored back into eight bits.
fn signed_saturating_operand(value: Value, ctx: &Context) -> String {
    let name = value.name(ctx);
    match value.is_signed_int(ctx) && value.scalar_ty(ctx).size(ctx) == 1 {
        true => format!("char({name})"),
        false => name.to_string(),
    }
}

unrolling!(SaturatingSAddOp);
metal_op_with_out!(SaturatingSAddOp, |op, ctx| {
    let lhs = signed_saturating_operand(op.lhs(ctx), ctx);
    let rhs = signed_saturating_operand(op.rhs(ctx), ctx);
    format!("addsat({lhs}, {rhs})")
});
unrolling!(SaturatingUAddOp);
metal_op_with_out!(SaturatingUAddOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("addsat({lhs}, {rhs})")
});

unrolling!(SaturatingSSubOp);
metal_op_with_out!(SaturatingSSubOp, |op, ctx| {
    let lhs = signed_saturating_operand(op.lhs(ctx), ctx);
    let rhs = signed_saturating_operand(op.rhs(ctx), ctx);
    format!("subsat({lhs}, {rhs})")
});
unrolling!(SaturatingUSubOp);
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
no_msl_bfloat!(FMinOp);

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
no_msl_bfloat!(FMaxOp);

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
    format!("rsqrt({lhs} * {lhs} + {rhs} * {rhs})")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        shared::operation::OpToCPP,
        target::{CtxTarget, Target},
    };
    use cubecl_core::ir::{ConstantValue, ElemType, IntKind};
    use pliron::{attribute::boxed_attr_cast, builtin::ops::ConstantOp};

    fn saturating_add(kind: IntKind) -> String {
        let mut ctx = Context::new();
        ctx.set_target(Target::Metal);
        let mut constant = |value| {
            let attr = ConstantValue::Int(value).as_attribute(&ctx, ElemType::Int(kind));
            let attr = boxed_attr_cast(attr).unwrap();
            ConstantOp::new(&mut ctx, attr).get_result(&ctx)
        };
        let (lhs, rhs) = (constant(100), constant(100));
        let names = [lhs.name(&ctx).to_string(), rhs.name(&ctx).to_string()];
        let op = SaturatingSAddOp::new(&mut ctx, lhs, rhs);
        let msl = OpToCPP::<Metal>::to_cpp(&op, &ctx);
        let expr = msl.split_once(" = ").unwrap().1;
        expr.replace(&names[0], "lhs").replace(&names[1], "rhs")
    }

    /// An i8 saturates at eight bits only when the call resolves to MSL's `char` overload;
    /// through `int8_t` it resolves to `int` and wraps on the store.
    #[test]
    fn an_i8_saturating_add_calls_the_char_overload() {
        assert_eq!(
            saturating_add(IntKind::I8),
            "addsat(char(lhs), char(rhs));\n"
        );
        assert_eq!(saturating_add(IntKind::I16), "addsat(lhs, rhs);\n");
    }
}
