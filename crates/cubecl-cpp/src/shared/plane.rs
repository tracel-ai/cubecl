use cubecl_core::{
    self as cubecl,
    ir::{dialect::plane, prelude::*},
    prelude::*,
};

use cubecl_core::prelude::polyfills::plane::{
    OpAdd, OpMax, OpMin, OpMul, plane_reduce, plane_reduce_exclusive, plane_reduce_inclusive,
};

use crate::{
    cuda::packed_ops::packable,
    shared::{lowering::LowerOp, shared_op_with_out, unroll::unrolling},
    target::{CtxTarget, Target},
};

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

#[cube_op(name = "cpp.activemask")]
#[result_ty(argument)]
struct ActiveMask {}
shared_op_with_out!(ActiveMask, |_, _| "__activemask()".into());

#[cube]
fn activemask<T: Int>() -> T {
    intrinsic!(|scope| {
        let mask = ActiveMask::new(scope.ctx_mut(), T::__expand_as_type(scope));
        scope.register_with_result(&mask).into()
    })
}

// Lowest active lane, requires a generic because HIP uses u64 and CUDA uses u32 for `__activemask()`
#[cube]
pub fn elect<T: Int>() -> bool {
    u32::cast_from(activemask::<T>().trailing_zeros()) == UNIT_POS_PLANE
}
