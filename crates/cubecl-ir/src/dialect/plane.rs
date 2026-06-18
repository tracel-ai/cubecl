use cubecl_macros_internal::cube_op;
use pliron::r#type::TypeHandle;

use crate::{
    attributes::IndexAttr,
    dialect::{ptr_value_ty, synchronization::SyncScope},
    interfaces::synchronizes,
    prelude::*,
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
            pub input: Value,
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
    pub input: Value,
}
synchronizes!(BallotOp, SyncScope::Plane);

fn ballot_ty(ctx: &Context) -> TypeHandle {
    let u32 = UIntType::get(ctx, 32);
    VectorType::get(ctx, u32.into(), 4).into()
}

#[cube_op(name = "plane.broadcast")]
#[result_ty(same_as = input)]
pub struct BroadcastOp {
    pub input: Value,
    pub lane: IndexAttr,
}
synchronizes!(BroadcastOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle")]
#[result_ty(same_as = input)]
pub struct ShuffleOp {
    pub input: Value,
    pub lane: Value,
}
synchronizes!(ShuffleOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle_xor")]
#[result_ty(same_as = input)]
pub struct ShuffleXorOp {
    pub input: Value,
    pub mask: Value,
}
synchronizes!(ShuffleXorOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle_up")]
#[result_ty(same_as = input)]
pub struct ShuffleUpOp {
    pub input: Value,
    pub delta: Value,
}
synchronizes!(ShuffleUpOp, SyncScope::Plane);

#[cube_op(name = "plane.shuffle_down")]
#[result_ty(same_as = input)]
pub struct ShuffleDownOp {
    pub input: Value,
    pub delta: Value,
}
synchronizes!(ShuffleDownOp, SyncScope::Plane);

#[cube_op(name = "plane.uniform_load")]
#[result_ty(from_inputs = ptr_value_ty)]
pub struct UniformLoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
}
synchronizes!(UniformLoadOp, SyncScope::Plane);
