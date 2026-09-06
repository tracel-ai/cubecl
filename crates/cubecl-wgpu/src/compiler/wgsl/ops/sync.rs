use cubecl_ir::dialect::synchronization::{SyncOp, SyncScope};

use crate::compiler::wgsl::to_wgsl::{keep_wgsl_ops_linked, wgsl_op};

keep_wgsl_ops_linked!(SyncOp);

wgsl_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane | SyncScope::Cube => "workgroupBarrier();\n".into(),
        SyncScope::Device => "storageBarrier();\n".into(),
        SyncScope::Unit => "".into(),
    }
});
