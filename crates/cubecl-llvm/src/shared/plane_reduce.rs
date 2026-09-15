//! Plane reductions and scans.

use crate::prelude::*;
use cubecl_core::{
    ir::dialect::plane,
    prelude::{
        polyfills::plane::{
            OpAdd, OpMax, OpMin, OpMul, plane_reduce, plane_reduce_exclusive,
            plane_reduce_inclusive,
        },
        *,
    },
};

define_scalar!(T);
define_size!(S);

macro_rules! lower_reduction {
    ($ty:ty, $reduce:ident, $op:ty $(, $args:expr)*) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                ctx.target().is_gpu()
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let input = self.input(scope.ctx());
                scope.register_value_type::<T, S>(input);
                vec![
                    $reduce::expand::<T, S, $op>(scope, input.into() $(, $args)*)
                        .read_value(scope),
                ]
            }
        }
    };
}

lower_reduction!(plane::ISumOp, plane_reduce, OpAdd);
lower_reduction!(plane::FSumOp, plane_reduce, OpAdd);
lower_reduction!(plane::IProdOp, plane_reduce, OpMul);
lower_reduction!(plane::FProdOp, plane_reduce, OpMul);
lower_reduction!(plane::SMinOp, plane_reduce, OpMin);
lower_reduction!(plane::UMinOp, plane_reduce, OpMin);
lower_reduction!(plane::FMinOp, plane_reduce, OpMin);
lower_reduction!(plane::SMaxOp, plane_reduce, OpMax);
lower_reduction!(plane::UMaxOp, plane_reduce, OpMax);
lower_reduction!(plane::FMaxOp, plane_reduce, OpMax);

lower_reduction!(plane::InclusiveISumOp, plane_reduce_inclusive, OpAdd);
lower_reduction!(plane::InclusiveFSumOp, plane_reduce_inclusive, OpAdd);
lower_reduction!(plane::InclusiveIProdOp, plane_reduce_inclusive, OpMul);
lower_reduction!(plane::InclusiveFProdOp, plane_reduce_inclusive, OpMul);

lower_reduction!(plane::ExclusiveISumOp, plane_reduce_exclusive, OpAdd, 0);
lower_reduction!(plane::ExclusiveFSumOp, plane_reduce_exclusive, OpAdd, 0);
lower_reduction!(plane::ExclusiveIProdOp, plane_reduce_exclusive, OpMul, 1);
lower_reduction!(plane::ExclusiveFProdOp, plane_reduce_exclusive, OpMul, 1);
