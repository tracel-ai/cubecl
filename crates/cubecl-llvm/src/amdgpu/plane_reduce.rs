//! The plane reductions and scans, as folds over the shuffles.
//!
//! The wavefront has no reduction instruction of its own, so each of these is a logarithmic
//! fold over [`plane`](super::plane)'s shuffles. The folds themselves are
//! [`cubecl_core::prelude::polyfills::plane`], shared with the C++ backends.

use cubecl_core::ir::Scope;
use cubecl_core::ir::dialect::plane;
use cubecl_core::ir::prelude::*;
use cubecl_core::prelude::polyfills::plane::{
    OpAdd, OpMax, OpMin, OpMul, plane_reduce, plane_reduce_exclusive, plane_reduce_inclusive,
};
use cubecl_core::prelude::*;

use crate::shared::polyfill::LowerOp;
use crate::target::{CtxTarget, LlvmTarget};

define_scalar!(T);
define_size!(S);

/// Lowers a plane reduction to the fold `$reduce` of `$op`.
macro_rules! lower_reduction {
    ($ty:ty, $reduce:ident, $op:ty $(, $args:expr)*) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                // The CPU has a plane of one unit, so its reductions are the value itself and
                // are lowered elsewhere.
                ctx.target() == LlvmTarget::AmdGpu
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
