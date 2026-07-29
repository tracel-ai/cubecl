use cubecl_core::{
    self as cubecl,
    frontend::polyfills::expm1,
    ir::{
        CanMaterialize, Pure, cube_op,
        dialect::{
            bitwise::*,
            general::ReinterpretCastOp,
            math::{Expm1Op, FAbsOp, TanhOp},
        },
        interfaces::TypedExt,
        prelude::op_traits,
    },
    prelude::*,
};
use pliron::value::Value;

use crate::{
    metal::metal_op_with_out,
    shared::{ty::TypeExtCPP, unary::lower_target_unop, unroll::unrolling},
    target::Metal,
};

metal_op_with_out!(FAbsOp, |op, ctx| {
    format!("abs({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(CountOnesOp, |op, ctx| {
    format!("popcount({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(ReverseBitsOp, |op, ctx| {
    format!("reverse_bits({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(LeadingZerosBitsOp, |op, ctx| {
    format!("clz({})", op.input(ctx).name(ctx))
});

unrolling!(TrailingZerosBitsOp);
metal_op_with_out!(TrailingZerosBitsOp, |op, ctx| {
    format!("ctz({})", op.input(ctx).name(ctx))
});

metal_op_with_out!(ReinterpretCastOp, |op, ctx| {
    let input = op.input(ctx);
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    if input.is_ptr(ctx) {
        format!("reinterpret_cast<{ty}>({})", input.name(ctx))
    } else {
        format!("reinterpret_cast<const thread {ty}&>({})", input.name(ctx))
    }
});

lower_target_unop!(Expm1Op, expm1, Metal);
lower_target_unop!(TanhOp, safe_tanh, Metal);

#[cube_op(name = "msl.tanh")]
#[result_ty(same_as = input)]
#[op_traits(Pure, CanMaterialize)]
pub struct MslTanhOp {
    pub input: Value,
}

unrolling!(MslTanhOp);
metal_op_with_out!(MslTanhOp, |op, ctx| {
    format!("tanh({})", op.input(ctx).name(ctx))
});

/// use the simple version because otherwise we'd get an infinite lowering loop
#[cube]
fn simple_tanh<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    intrinsic!(|scope| {
        let input = input.read_value(scope);
        let tanh = MslTanhOp::new(scope.ctx_mut(), input);
        scope.register_with_result(&tanh).into()
    })
}

#[cube]
fn safe_tanh<T: Float, N: Size>(x: Vector<T, N>) -> Vector<T, N> {
    let threshold = Vector::new(T::new(43.0_f32));
    select(x > threshold, Vector::one(), simple_tanh(x))
}
