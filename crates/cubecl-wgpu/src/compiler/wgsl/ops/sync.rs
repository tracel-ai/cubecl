use cubecl_ir::dialect::synchronization::{SyncOp, SyncScope};

use crate::compiler::wgsl::to_wgsl::wgsl_op;

wgsl_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane | SyncScope::Cube => "workgroupBarrier();\n".into(),
        // The storage half is all WGSL can say across workgroups, and the workgroup
        // barrier is the cube scope this one contains.
        SyncScope::Device => "storageBarrier();\nworkgroupBarrier();\n".into(),
        SyncScope::Unit => "".into(),
    }
});
