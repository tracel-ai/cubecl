use cubecl_core::ir::{dialect::synchronization::SyncAsyncProxyOp, prelude::*};

use crate::{shared::signature::op_includes, target::Cuda};

macro_rules! cuda_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Cuda> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use cuda_op;

macro_rules! cuda_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Cuda> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                use cubecl_core::ir::prelude::*;
                use $crate::shared::CppValue;
                let op = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl);
                let out = self.get_result(ctx).fmt_left(ctx);
                format!("{out} = {op};\n")
            }
        }
    };
}
pub(super) use cuda_op_with_out;

macro_rules! ptx_with_out {
    ($ty: ty, $ptx: expr, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::shared::lowering::LowerOp<$crate::target::Cuda> for $ty {
            fn should_lower(&self, ctx: &pliron::context::Context) -> bool {
                $crate::shared::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }
            fn lower(&self, scope: &cubecl_core::ir::Scope) -> Vec<pliron::value::Value> {
                use cubecl_core::ir::dialect::base::OperationPtrExt;
                use pliron::{op::Op, r#type::Typed};
                let ctx = scope.ctx_mut();
                let ptx = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $ptx);
                let op = $crate::cuda::ptx::InlinePtxOp::new(
                    ctx,
                    Some(self.get_result(ctx).get_type(ctx)),
                    ptx,
                    self.get_operation().operands(ctx),
                );
                scope.register(&op);
                vec![op.result(ctx).unwrap()]
            }
        }
    };
    ($ty: ty, $ptx: expr) => {
        ptx_with_out!($ty, $ptx, |_, _| true);
    };
}
pub(super) use ptx_with_out;

op_includes!(Cuda, [SyncAsyncProxyOp] => "cuda/barrier");

cuda_op!(SyncAsyncProxyOp, |_, _| {
    "cuda::device::experimental::fence_proxy_async_shared_cta();".into()
});
