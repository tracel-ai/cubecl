use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncPlaneStorageOp, SyncScope};

macro_rules! opencl_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::OpenCl> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use opencl_op;

macro_rules! opencl_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::OpenCl> for $ty {
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
pub(super) use opencl_op_with_out;

/// An op whose value is `function(operands...)`, its operands named by their accessors.
macro_rules! call {
    ($ty: ty, $function: literal, $($operand: ident),+) => {
        $crate::opencl::opencl_op_with_out!($ty, |op, ctx| {
            let operands = [$(op.$operand(ctx).name(ctx).to_string()),+];
            format!("{}({})", $function, operands.join(", "))
        });
    };
}
pub(super) use call;

// Plane and cube syncs order shared memory, as on Metal and WGSL; the device sync (`sync_storage`)
// orders global memory too, with device-scope fences where the device has them (`sync_device` in
// the header). The runtime shares no memory between devices, so a system sync is a device one.
opencl_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane => "sub_group_barrier(CLK_LOCAL_MEM_FENCE);\n",
        SyncScope::Cube => "barrier(CLK_LOCAL_MEM_FENCE);\n",
        SyncScope::Device | SyncScope::System => "sync_device();\n",
        SyncScope::Unit => "",
    }
    .into()
});

// A plane barrier ordering global memory too: the subgroup's storage writes, for its own units.
opencl_op!(SyncPlaneStorageOp, |_, _| {
    "sub_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n".into()
});
