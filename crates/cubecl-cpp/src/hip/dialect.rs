macro_rules! hip_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Hip> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncScope};
pub(super) use hip_op;

macro_rules! hip_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Hip> for $ty {
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
pub(super) use hip_op_with_out;

hip_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane => {
            // HIP has no `__syncwarp`. AMD wavefronts execute in lockstep, so the
            // execution half of the sync is a compiler scheduling barrier; the
            // wavefront-scope fence supplies the memory ordering `__syncwarp`
            // carries on CUDA (LDS/global writes by other lanes of the wave are
            // visible past the sync).

            "
__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, \"wavefront\");
__builtin_amdgcn_wave_barrier();\n"
        }
        SyncScope::Cube | SyncScope::Device => "__syncthreads();\n",
        SyncScope::Unit => "",
    }
    .into()
});
