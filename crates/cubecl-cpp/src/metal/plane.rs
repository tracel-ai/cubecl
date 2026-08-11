use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::ir::{cube_op, dialect::plane::*};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    derive::op_interface_impl,
    value::Value,
};

use crate::{
    metal::metal_op_with_out,
    shared::{lowering::LowerOp, unroll::unrolling},
    target::Metal,
};

metal_op_with_out!(BroadcastOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).0;
    format!("simd_shuffle({val}, {lane});")
});

metal_op_with_out!(ShuffleOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).name(ctx);
    format!("simd_shuffle({val}, {lane});")
});

metal_op_with_out!(ShuffleXorOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let mask = op.mask(ctx).name(ctx);
    format!("simd_shuffle_xor({val}, {mask});")
});

metal_op_with_out!(ShuffleUpOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("simd_shuffle_up({val}, {delta});")
});

metal_op_with_out!(ShuffleDownOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let delta = op.delta(ctx).name(ctx);
    format!("simd_shuffle_down({val}, {delta});")
});

metal_op_with_out!(AllOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_all({val});")
});

metal_op_with_out!(AnyOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_any({val});")
});

#[cube_op(name = "msl.ballot")]
#[result_ty(fixed = IntegerType::get(ctx, 64, Signedness::Unsigned).to_handle())]
pub struct MslBallotOp {
    input: Value,
}

metal_op_with_out!(MslBallotOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("uint64_t(simd_ballot({val}));")
});

#[cube]
fn msl_ballot(value: bool) -> u64 {
    intrinsic!(|scope| {
        let value = value.read_value(scope);
        let ballot = MslBallotOp::new(scope.ctx_mut(), value);
        scope.register_with_result(&ballot).into()
    })
}

#[cube]
fn ballot(value: bool) -> Vector<u32, Const<4>> {
    let mut out = Vector::<u64, Const<2>>::zero();
    out.insert(0usize, msl_ballot(value));
    Vector::reinterpret(out)
}

#[op_interface_impl]
impl LowerOp<Metal> for BallotOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let value = self.input(scope.ctx()).into();
        vec![ballot::expand(scope, value).read_value(scope)]
    }
}

unrolling!(ISumOp);
metal_op_with_out!(ISumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_sum({val})")
});
unrolling!(FSumOp);
metal_op_with_out!(FSumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_sum({val})")
});

unrolling!(IProdOp);
metal_op_with_out!(IProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_product({val})")
});
unrolling!(FProdOp);
metal_op_with_out!(FProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_product({val})")
});

unrolling!(SMinOp);
metal_op_with_out!(SMinOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_min({val})")
});
unrolling!(UMinOp);
metal_op_with_out!(UMinOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_min({val})")
});
unrolling!(FMinOp);
metal_op_with_out!(FMinOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_min({val})")
});

unrolling!(SMaxOp);
metal_op_with_out!(SMaxOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_max({val})")
});
unrolling!(UMaxOp);
metal_op_with_out!(UMaxOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_max({val})")
});
unrolling!(FMaxOp);
metal_op_with_out!(FMaxOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_max({val})")
});

unrolling!(InclusiveISumOp);
metal_op_with_out!(InclusiveISumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_inclusive_sum({val})")
});
unrolling!(InclusiveFSumOp);
metal_op_with_out!(InclusiveFSumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_inclusive_sum({val})")
});

unrolling!(InclusiveIProdOp);
metal_op_with_out!(InclusiveIProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_inclusive_product({val})")
});
unrolling!(InclusiveFProdOp);
metal_op_with_out!(InclusiveFProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_inclusive_product({val})")
});

unrolling!(ExclusiveISumOp);
metal_op_with_out!(ExclusiveISumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_exclusive_sum({val})")
});
unrolling!(ExclusiveFSumOp);
metal_op_with_out!(ExclusiveFSumOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_exclusive_sum({val})")
});

unrolling!(ExclusiveIProdOp);
metal_op_with_out!(ExclusiveIProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_exclusive_product({val})")
});
unrolling!(ExclusiveFProdOp);
metal_op_with_out!(ExclusiveFProdOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    format!("simd_prefix_exclusive_product({val})")
});
