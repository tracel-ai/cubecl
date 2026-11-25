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
use cubecl_core::ir::dialect::synchronization::{
    FenceOp, MemoryOrder, ScheduleBarrierOp, SyncOp, SyncPlaneStorageOp, SyncScope,
};
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
            // wavefront-scope fence makes LDS writes by other lanes of the wave
            // visible past the sync.

            "
__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, \"wavefront\", \"local\");
__builtin_amdgcn_wave_barrier();\n"
        }
        // `__syncthreads` fences global memory as well, so the barrier and its fences are written
        // out.
        SyncScope::Cube => {
            "
__builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\", \"local\");
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\", \"local\");\n"
        }
        // As on CUDA: `__syncthreads` is the block's own ordering, and the fence is what makes
        // this block's writes visible to the others.
        SyncScope::Device => "__threadfence();\n__syncthreads();\n",
        // The same at the system's scope, which reaches the host and peer devices.
        SyncScope::System => "__threadfence_system();\n__syncthreads();\n__threadfence_system();\n",
        SyncScope::Unit => "",
    }
    .into()
});

hip_op!(FenceOp, |op, ctx| {
    let order = match op.order(ctx).0 {
        MemoryOrder::Acquire => "__ATOMIC_ACQUIRE",
        MemoryOrder::Release => "__ATOMIC_RELEASE",
        MemoryOrder::AcqRel => "__ATOMIC_ACQ_REL",
    };
    let scope = match op.scope(ctx).0 {
        SyncScope::Plane => "wavefront",
        SyncScope::Cube => "workgroup",
        SyncScope::Device => "agent",
        SyncScope::System => "",
        SyncScope::Unit => unreachable!("a unit fence is dropped at expansion"),
    };
    format!("__builtin_amdgcn_fence({order}, \"{scope}\");\n")
});

// Mask 0: the backend's schedulers move no instruction across it.
hip_op!(ScheduleBarrierOp, |_, _| {
    "__builtin_amdgcn_sched_barrier(0);\n".into()
});

// The plane barrier with its fence over every address space, global included.
hip_op!(SyncPlaneStorageOp, |_, _| {
    "
__builtin_amdgcn_fence(__ATOMIC_ACQ_REL, \"wavefront\");
__builtin_amdgcn_wave_barrier();\n"
        .into()
});
