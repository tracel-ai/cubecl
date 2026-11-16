use cubecl_core::{
    frontend::polyfills::powi,
    ir::{
        dialect::{
            cmp::{FClampOp, FMaxOp, FMinOp, SClampOp, UClampOp},
            math::{PowiOp, RhypotOp, SMulHiOp, UMulHiOp},
            memory::IndexOp,
        },
        interfaces::TypedExt,
        prelude::*,
        types::{ArrayType, PointerType},
    },
    prelude::*,
};

use crate::{
    opencl::{call, opencl_op_with_out},
    shared::lowering::LowerOp,
    target::OpenCl,
};

// `pow` (and `pown`) may be 16 ulp off: a float's integer power is multiplied out, as on Metal.
#[op_interface_impl]
impl LowerOp<OpenCl> for PowiOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        self.lhs(ctx).scalar_ty(ctx).is_float(ctx)
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let lhs = self.lhs(scope.ctx());
        let rhs = self.rhs(scope.ctx());
        scope.register_value_type::<T, S>(lhs);
        vec![powi::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)]
    }
}

call!(SMulHiOp, "mul_hi", lhs, rhs);
call!(UMulHiOp, "mul_hi", lhs, rhs);
call!(FMinOp, "fmin", lhs, rhs);
call!(FMaxOp, "fmax", lhs, rhs);
call!(SClampOp, "clamp", input, min, max);
call!(UClampOp, "clamp", input, min, max);
opencl_op_with_out!(FClampOp, |op, ctx| {
    let (input, min, max) = (
        op.input(ctx).name(ctx),
        op.min(ctx).name(ctx),
        op.max(ctx).name(ctx),
    );
    format!("fmax(fmin({input}, {max}), {min})")
});

opencl_op_with_out!(RhypotOp, |op, ctx| {
    format!(
        "1 / hypot({}, {})",
        op.lhs(ctx).name(ctx),
        op.rhs(ctx).name(ctx)
    )
});

// Arrays are structs around a C array.
opencl_op_with_out!(IndexOp, |op, ctx| {
    let base = op.base(ctx);
    let index = op.index(ctx).name(ctx);
    let base_ty = base.get_type(ctx).deref(ctx);
    let name = base.name(ctx);
    match base_ty.downcast_ref::<PointerType>() {
        Some(PointerType { inner, .. }) if inner.deref(ctx).is::<ArrayType>() => {
            format!("&({name}->data[{index}])")
        }
        _ => format!("&{name}[{index}]"),
    }
});
