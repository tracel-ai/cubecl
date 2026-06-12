use cubecl_macros_internal::cube_op;

use crate::{
    attributes::IndexAttr,
    dialect::{ptr_value_ty, synchronization::SyncScope},
    interfaces::synchronizes,
    pliron::prelude::*,
    types::{
        VectorType,
        scalar::{BoolType, UIntType},
    },
};

#[cube_op(name = "plane.elect")]
#[result_ty(fixed = BoolType::get(ctx).into())]
pub struct ElectOp {}
synchronizes!(ElectOp, SyncScope::Plane);

macro_rules! unary_plane_op {
    ($name: literal, $ty: ident) => {
        #[cube_op(name = $name)]
        #[result_ty(same_as = input)]
        pub struct $ty {
            input: Value,
        }
        synchronizes!($ty, SyncScope::Plane);
    };
}

unary_plane_op!("plane.all", AllOp);
unary_plane_op!("plane.any", AnyOp);
unary_plane_op!("plane.sum", SumOp);
unary_plane_op!("plane.inclusive_sum", InclusiveSumOp);
unary_plane_op!("plane.exclusive_sum", ExclusiveSumOp);
unary_plane_op!("plane.prod", ProdOp);
unary_plane_op!("plane.inclusive_prod", InclusiveProdOp);
unary_plane_op!("plane.exclusive_prod", ExclusiveProdOp);
unary_plane_op!("plane.min", MinOp);
unary_plane_op!("plane.max", MaxOp);

#[cube_op(name = "plane.ballot")]
#[result_ty(fixed = ballot_ty(ctx))]
pub struct BallotOp {
    input: Value,
}
synchronizes!(BallotOp, SyncScope::Plane);

fn ballot_ty(ctx: &mut Context) -> Ptr<TypeObj> {
    let u32 = UIntType::get(ctx, 32);
    VectorType::get(ctx, u32.into(), 4).into()
}

#[cube_op(name = "plane.broadcast")]
#[result_ty(same_as = input)]
pub struct BroadcastOp {
    input: Value,
    index: IndexAttr,
}
synchronizes!(BroadcastOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle")]
#[result_ty(same_as = input)]
pub struct ShuffleOp {
    input: Value,
    lane: Value,
}
synchronizes!(ShuffleOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle_xor")]
#[result_ty(same_as = input)]
pub struct ShuffleXorOp {
    input: Value,
    mask: Value,
}
synchronizes!(ShuffleXorOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle_up")]
#[result_ty(same_as = input)]
pub struct ShuffleUpOp {
    input: Value,
    delta: Value,
}
synchronizes!(ShuffleUpOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle_down")]
#[result_ty(same_as = input)]
pub struct ShuffleDownOp {
    input: Value,
    delta: Value,
}
synchronizes!(ShuffleDownOp, SyncScope::Plane);

#[cube_op(name = "plane.uniform_load")]
#[result_ty(from_inputs = ptr_value_ty)]
pub struct UniformLoadOp {
    #[operand(ptr_read)]
    ptr: Value,
}
synchronizes!(UniformLoadOp, SyncScope::Plane);
