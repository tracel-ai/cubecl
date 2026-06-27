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
impl<T: Scalar + PartialOrd, N: Size> PlaneOp<T, N> for OpMin {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        min(lhs, rhs)
    }
}
#[cube]
impl<T: Scalar + PartialOrd, N: Size> PlaneOp<T, N> for OpMax {
    fn apply(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
        max(lhs, rhs)
    }
}

#[cube]
pub fn plane_reduce<T: Scalar, N: Size, Op: PlaneOp<T, N>>(val: Vector<T, N>) -> Vector<T, N> {
    let mut acc = val;
    let mut offset = 1;
    while offset < PLANE_DIM {
        acc = Op::apply(acc, plane_shuffle_xor(acc, offset));
        offset *= 2;
    }
    acc
}

#[cube]
pub fn plane_reduce_inclusive<T: Scalar, N: Size, Op: PlaneOp<T, N>>(
    val: Vector<T, N>,
) -> Vector<T, N> {
    let mut acc = val;
    let mut offset = 1;
    while offset < PLANE_DIM {
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

lower_unop!(plane::SumOp, plane_reduce, OpAdd);
lower_unop!(plane::ProdOp, plane_reduce, OpMul);
lower_unop!(plane::MinOp, plane_reduce, OpMin);
lower_unop!(plane::MaxOp, plane_reduce, OpMax);

lower_unop!(plane::InclusiveSumOp, plane_reduce_inclusive, OpAdd);
lower_unop!(plane::InclusiveProdOp, plane_reduce_inclusive, OpMul);

lower_unop!(plane::ExclusiveSumOp, plane_reduce_exclusive, OpAdd, 0);
lower_unop!(plane::ExclusiveProdOp, plane_reduce_exclusive, OpMul, 1);

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
