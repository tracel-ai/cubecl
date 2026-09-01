use cubecl_macros_internal::{cube_op, op_traits};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    r#type::TypeHandle,
};

use crate::{
    CanMaterialize, NoMemoryEffect,
    attributes::IndexAttr,
    dialect::{ptr_value_ty, synchronization::SyncScope},
    interfaces::{
        TriviallyUnrollable, synchronizes,
        uniformity::{UniformOpInterface, Uniformity},
    },
    prelude::*,
    types::{VectorType, scalar::BoolType},
};

#[cube_op(name = "plane.elect")]
#[result_ty(fixed = BoolType::get(ctx).into())]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ElectOp {}
synchronizes!(ElectOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for ElectOp {
    fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        Uniformity::None
    }
}

macro_rules! unary_plane_op {
    ($name: literal, $ty: ident) => {
        #[cube_op(name = $name)]
        #[result_ty(same_as = input)]
        #[op_interfaces(TriviallyUnrollable)]
        #[op_traits(CanMaterialize, NoMemoryEffect)]
        pub struct $ty {
            pub input: Value,
        }
        synchronizes!($ty, SyncScope::Plane);

        #[op_interface_impl]
        impl UniformOpInterface for $ty {
            fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
                operands[0].max(Uniformity::Plane)
            }
        }
    };
}

macro_rules! nonuniform_unary_plane_op {
    ($name: literal, $ty: ident) => {
        #[cube_op(name = $name)]
        #[result_ty(same_as = input)]
        #[op_interfaces(TriviallyUnrollable)]
        #[op_traits(CanMaterialize, NoMemoryEffect)]
        pub struct $ty {
            pub input: Value,
        }
        synchronizes!($ty, SyncScope::Plane);

        #[op_interface_impl]
        impl UniformOpInterface for $ty {
            fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
                Uniformity::None
            }
        }
    };
}

unary_plane_op!("plane.all", AllOp);
unary_plane_op!("plane.any", AnyOp);
unary_plane_op!("plane.i_sum", ISumOp);
unary_plane_op!("plane.f_sum", FSumOp);
unary_plane_op!("plane.inclusive_i_sum", InclusiveISumOp);
unary_plane_op!("plane.inclusive_f_sum", InclusiveFSumOp);
unary_plane_op!("plane.exclusive_i_sum", ExclusiveISumOp);
unary_plane_op!("plane.exclusive_f_sum", ExclusiveFSumOp);
unary_plane_op!("plane.i_prod", IProdOp);
unary_plane_op!("plane.f_prod", FProdOp);
nonuniform_unary_plane_op!("plane.inclusive_i_prod", InclusiveIProdOp);
nonuniform_unary_plane_op!("plane.inclusive_f_prod", InclusiveFProdOp);
nonuniform_unary_plane_op!("plane.exclusive_i_prod", ExclusiveIProdOp);
nonuniform_unary_plane_op!("plane.exclusive_f_prod", ExclusiveFProdOp);
unary_plane_op!("plane.s_min", SMinOp);
unary_plane_op!("plane.u_min", UMinOp);
unary_plane_op!("plane.f_min", FMinOp);
unary_plane_op!("plane.s_max", SMaxOp);
unary_plane_op!("plane.u_max", UMaxOp);
unary_plane_op!("plane.f_max", FMaxOp);

#[cube_op(name = "plane.ballot")]
#[result_ty(fixed = ballot_ty(ctx))]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct BallotOp {
    pub input: Value,
}
synchronizes!(BallotOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for BallotOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0].max(Uniformity::Plane)
    }
}

fn ballot_ty(ctx: &Context) -> TypeHandle {
    let u32 = IntegerType::get(ctx, 32, Signedness::Unsigned);
    VectorType::get(ctx, u32.into(), 4).into()
}

#[cube_op(name = "plane.broadcast")]
#[result_ty(same_as = input)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct BroadcastOp {
    pub input: Value,
    pub lane: IndexAttr,
}
synchronizes!(BroadcastOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for BroadcastOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0].max(Uniformity::Plane)
    }
}

#[cube_op(name = "plane.shuffle")]
#[result_ty(same_as = input)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ShuffleOp {
    pub input: Value,
    pub lane: Value,
}
synchronizes!(ShuffleOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for ShuffleOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0]
    }
}

#[cube_op(name = "plane.shuffle_xor")]
#[result_ty(same_as = input)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ShuffleXorOp {
    pub input: Value,
    pub mask: Value,
}
synchronizes!(ShuffleXorOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for ShuffleXorOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0]
    }
}

#[cube_op(name = "plane.shuffle_up")]
#[result_ty(same_as = input)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ShuffleUpOp {
    pub input: Value,
    pub delta: Value,
}
synchronizes!(ShuffleUpOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for ShuffleUpOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0]
    }
}

#[cube_op(name = "plane.shuffle_down")]
#[result_ty(same_as = input)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize, NoMemoryEffect)]
pub struct ShuffleDownOp {
    pub input: Value,
    pub delta: Value,
}
synchronizes!(ShuffleDownOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for ShuffleDownOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0]
    }
}

#[cube_op(name = "plane.uniform_load")]
#[result_ty(from_inputs = ptr_value_ty)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize)]
pub struct UniformLoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
}
synchronizes!(UniformLoadOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for UniformLoadOp {
    fn uniformity(&self, _ctx: &Context, operands: &[Uniformity]) -> Uniformity {
        operands[0].max(Uniformity::Plane)
    }
}

#[cube_op(name = "plane.atomic_uniform_load")]
#[result_ty(from_inputs = ptr_value_ty)]
#[op_interfaces(TriviallyUnrollable)]
#[op_traits(CanMaterialize)]
pub struct AtomicUniformLoadOp {
    #[operand(ptr_read)]
    pub ptr: Value,
}
synchronizes!(AtomicUniformLoadOp, SyncScope::Plane);

#[op_interface_impl]
impl UniformOpInterface for AtomicUniformLoadOp {
    fn uniformity(&self, _ctx: &Context, _operands: &[Uniformity]) -> Uniformity {
        Uniformity::Plane
    }
}
