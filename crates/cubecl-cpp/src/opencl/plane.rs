use cubecl_core::ir::{dialect::plane::*, prelude::*};

use crate::{
    opencl::{call, opencl_op_with_out},
    shared::ty::TypeExtCPP,
};

// Plane operations act on the units that reach them (as CUDA's active mask and SPIR-V's
// `GroupNonUniform*`): the `cl_khr_subgroup_*` non-uniform forms (ballot, shuffle,
// shuffle_relative, non_uniform_vote, non_uniform_arithmetic, which also take 8- and 16-bit
// integers); vectors run lane by lane.

// The lane is a constant, so uniform across the plane as the non-uniform broadcast requires.
opencl_op_with_out!(BroadcastOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let lane = op.lane(ctx).0;
    format!("sub_group_non_uniform_broadcast({val}, {lane})")
});

call!(ShuffleOp, "sub_group_shuffle", input, lane);

call!(ShuffleXorOp, "sub_group_shuffle_xor", input, mask);

call!(ShuffleUpOp, "sub_group_shuffle_up", input, delta);

call!(ShuffleDownOp, "sub_group_shuffle_down", input, delta);

opencl_op_with_out!(ElectOp, |_, _| "sub_group_elect()".into());

opencl_op_with_out!(BallotOp, |op, ctx| {
    let val = op.input(ctx).name(ctx);
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!(
        "({{ uint4 ballot = sub_group_ballot({val}); ({ty}){{ ballot.x, ballot.y, ballot.z, ballot.w }}; }})"
    )
});

call!(AllOp, "sub_group_non_uniform_all", input);
call!(AnyOp, "sub_group_non_uniform_any", input);
call!(ISumOp, "sub_group_non_uniform_reduce_add", input);
call!(FSumOp, "sub_group_non_uniform_reduce_add", input);
call!(IProdOp, "sub_group_non_uniform_reduce_mul", input);
call!(FProdOp, "sub_group_non_uniform_reduce_mul", input);
call!(SMinOp, "sub_group_non_uniform_reduce_min", input);
call!(UMinOp, "sub_group_non_uniform_reduce_min", input);
call!(FMinOp, "sub_group_non_uniform_reduce_min", input);
call!(SMaxOp, "sub_group_non_uniform_reduce_max", input);
call!(UMaxOp, "sub_group_non_uniform_reduce_max", input);
call!(FMaxOp, "sub_group_non_uniform_reduce_max", input);
call!(
    InclusiveISumOp,
    "sub_group_non_uniform_scan_inclusive_add",
    input
);
call!(
    InclusiveFSumOp,
    "sub_group_non_uniform_scan_inclusive_add",
    input
);
call!(
    InclusiveIProdOp,
    "sub_group_non_uniform_scan_inclusive_mul",
    input
);
call!(
    InclusiveFProdOp,
    "sub_group_non_uniform_scan_inclusive_mul",
    input
);
call!(
    ExclusiveISumOp,
    "sub_group_non_uniform_scan_exclusive_add",
    input
);
call!(
    ExclusiveFSumOp,
    "sub_group_non_uniform_scan_exclusive_add",
    input
);
call!(
    ExclusiveIProdOp,
    "sub_group_non_uniform_scan_exclusive_mul",
    input
);
call!(
    ExclusiveFProdOp,
    "sub_group_non_uniform_scan_exclusive_mul",
    input
);
