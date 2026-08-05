use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::ir::{cube_op, dialect::plane::*};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    derive::op_interface_impl,
    value::Value,
};

use crate::{hip::hip_op_with_out, shared::lowering::LowerOp, target::Hip};

hip_op_with_out!(BroadcastOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).0;
    format!("__shfl({val}, {lane});")
});

hip_op_with_out!(ShuffleOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).name(ctx);
    format!("__shfl({val}, {lane});")
});

hip_op_with_out!(ShuffleXorOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let mask = op.mask(ctx).name(ctx);
    format!("__shfl_xor({val}, {mask});")
});

hip_op_with_out!(ShuffleUpOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("__shfl_up({val}, {delta});")
});

hip_op_with_out!(ShuffleDownOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("__shfl_down({val}, {delta});")
});

hip_op_with_out!(AllOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("static_cast<bool>(__all({val}));")
});

hip_op_with_out!(AnyOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("static_cast<bool>(__any({val}));")
});

#[cube_op(name = "hip.ballot")]
#[result_ty(fixed = IntegerType::get(ctx, 64, Signedness::Unsigned).to_handle())]
pub struct HipBallotOp {
    input: Value,
}

hip_op_with_out!(HipBallotOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("__ballot({val});")
});

#[cube]
fn hip_ballot(value: bool) -> u64 {
    intrinsic!(|scope| {
        let value = value.read_value(scope);
        let ballot = HipBallotOp::new(scope.ctx_mut(), value);
        scope.register_with_result(&ballot).into()
    })
}

/// Unlike CUDA's 32 bit `__ballot_sync`, HIP's `__ballot` returns a 64 bit mask so it can cover
/// wave64. It has to be split across two of the result's 32 bit lanes instead of narrowed into one.
#[cube]
fn ballot(value: bool) -> Vector<u32, Const<4>> {
    let mut out = Vector::<u64, Const<2>>::zero();
    out.insert(0usize, hip_ballot(value));
    Vector::reinterpret(out)
}

#[op_interface_impl]
impl LowerOp<Hip> for BallotOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let value = self.input(scope.ctx()).into();
        vec![ballot::expand(scope, value).read_value(scope)]
    }
}
