use cubecl_core::{
    self as cubecl,
    ir::{dialect::plane, prelude::*},
    prelude::*,
};

use crate::{
    cuda::packed_ops::packable,
    shared::{lowering::LowerOp, unroll::unrolling},
    target::{CtxTarget, Target},
};

#[cube]
pub trait PlaneOp<T: Scalar, N: Size> {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N>;
}

struct OpAdd;
struct OpMul;
struct OpMin;
struct OpMax;

#[cube]
impl<T: Scalar + CubeAdd, N: Size> PlaneOp<T, N> for OpAdd {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs + rhs
    }
}
#[cube]
impl<T: Scalar + CubeMul, N: Size> PlaneOp<T, N> for OpMul {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        lhs * rhs
    }
}
#[cube]
impl<T: Scalar + CubePartialOrd, N: Size> PlaneOp<T, N> for OpMin {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        min(lhs, rhs)
    }
}
#[cube]
impl<T: Scalar + CubePartialOrd, N: Size> PlaneOp<T, N> for OpMax {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        max(lhs, rhs)
    }
}

/// The number of units that actually take part in a plane operation.
///
/// A cube smaller than the plane leaves the upper lanes of the plane inactive. The butterfly
/// folds below must not reach past the last active lane: shuffling from a lane that isn't
/// running returns an unspecified value (the CUDA shuffles are masked with `__activemask()`,
/// so this reads as garbage rather than hanging), and folding that garbage into the
/// accumulator corrupts the result. `plane_sum` and `plane_max` happen to survive it — the
/// garbage reads as zero, which is the identity for one and below every input for the other —
/// but `plane_min` and `plane_prod` do not.
#[cube]
fn plane_dim_checked() -> u32 {
    min(PLANE_DIM, CUBE_DIM)
}

#[cube]
pub fn plane_reduce<T: Scalar, N: Size, Op: PlaneOp<T, N>>(val: Vector<T, N>) -> Vector<T, N> {
    let plane_dim = plane_dim_checked();
    let mut acc = val;
    let mut offset = 1;
    while offset < plane_dim {
        acc = Op::apply(acc, plane_shuffle_xor(acc, offset));
        offset *= 2;
    }
    acc
}

#[cube]
pub fn plane_reduce_inclusive<T: Scalar, N: Size, Op: PlaneOp<T, N>>(
    val: Vector<T, N>,
) -> Vector<T, N> {
    let plane_dim = plane_dim_checked();
    let mut acc = val;
    let mut offset = 1;
    while offset < plane_dim {
        let tmp = Op::apply(acc, plane_shuffle_up(acc, offset));
        if UNIT_POS_PLANE >= offset {
            acc = tmp;
        }
        offset *= 2;
    }
    acc
}

#[cube]
pub fn plane_reduce_exclusive<T: Numeric, N: Size, Op: PlaneOp<T, N>>(
    val: Vector<T, N>,
    #[comptime] default: i64,
) -> Vector<T, N> {
    let inclusive = plane_reduce_inclusive::<T, N, Op>(val);
    let shfl = plane_shuffle_up(inclusive, 1);
    select(UNIT_POS_PLANE == 0, Vector::new(T::from_int(default)), shfl)
}

#[cube]
pub fn elect() -> bool {
    UNIT_POS_PLANE == 0
}

/// Fallback lowering for `plane.elect`, for every target without a native election.
///
/// CUDA lowers this to PTX `elect.sync`, but only from Hopper on — `should_lower` there is
/// gated on `supports_features.elect_sync`, which the runtime sets only for `arch >= 90`.
/// Metal and HIP have no lowering at all. Without a fallback the op survives to the emitter
/// and fails as an `UnsupportedOp`, so this must stay unconditional: the target-specific
/// pass runs before this one and already claims the op wherever a native election exists.
///
/// Note this elects lane 0 rather than the lowest *active* lane, which is what `elect.sync`
/// gives you. The two agree whenever the plane is converged, which is the case for every
/// current caller; they diverge only under non-uniform control flow.
#[op_interface_impl]
impl LowerOp for plane::ElectOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        vec![elect::expand(scope).read_value(scope)]
    }
}

define_scalar!(T);
define_size!(S);

macro_rules! lower_unop {
    ($ty: ty, $reduce: ident, $op: ty $(,$args: expr)*) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                ctx.target() != Target::Metal
            }
            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let input = self.input(scope.ctx());
                scope.register_value_type::<T, S>(input);
                vec![$reduce::expand::<T, S, $op>(scope, input.into(), $($args),*).read_value(scope)]
            }
        }
    };
}

lower_unop!(plane::ISumOp, plane_reduce, OpAdd);
lower_unop!(plane::FSumOp, plane_reduce, OpAdd);
lower_unop!(plane::IProdOp, plane_reduce, OpMul);
lower_unop!(plane::FProdOp, plane_reduce, OpMul);
lower_unop!(plane::SMinOp, plane_reduce, OpMin);
lower_unop!(plane::UMinOp, plane_reduce, OpMin);
lower_unop!(plane::FMinOp, plane_reduce, OpMin);
lower_unop!(plane::SMaxOp, plane_reduce, OpMax);
lower_unop!(plane::UMaxOp, plane_reduce, OpMax);
lower_unop!(plane::FMaxOp, plane_reduce, OpMax);

lower_unop!(plane::InclusiveISumOp, plane_reduce_inclusive, OpAdd);
lower_unop!(plane::InclusiveFSumOp, plane_reduce_inclusive, OpAdd);
lower_unop!(plane::InclusiveIProdOp, plane_reduce_inclusive, OpMul);
lower_unop!(plane::InclusiveFProdOp, plane_reduce_inclusive, OpMul);

lower_unop!(plane::ExclusiveISumOp, plane_reduce_exclusive, OpAdd, 0);
lower_unop!(plane::ExclusiveFSumOp, plane_reduce_exclusive, OpAdd, 0);
lower_unop!(plane::ExclusiveIProdOp, plane_reduce_exclusive, OpMul, 1);
lower_unop!(plane::ExclusiveFProdOp, plane_reduce_exclusive, OpMul, 1);

unrolling!(plane::BroadcastOp);
packable!(plane::BroadcastOp);

unrolling!(plane::ShuffleOp);
packable!(plane::ShuffleOp);

unrolling!(plane::ShuffleXorOp);
packable!(plane::ShuffleXorOp);

unrolling!(plane::ShuffleUpOp);
packable!(plane::ShuffleUpOp);

unrolling!(plane::ShuffleDownOp);
packable!(plane::ShuffleDownOp);

unrolling!(plane::AllOp);
unrolling!(plane::AnyOp);
