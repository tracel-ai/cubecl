use cubecl_ir::dialect::synchronization::{SyncOp, SyncPlaneStorageOp, SyncScope};

use crate::compiler::wgsl::to_wgsl::{wasm_inventory_root, wgsl_op};

wasm_inventory_root!(SyncOp);

wgsl_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane | SyncScope::Cube => "workgroupBarrier();\n".into(),
        // The storage half is all WGSL can say across workgroups, and the workgroup
        // barrier is the cube scope this one contains.
        SyncScope::Device => "storageBarrier();\nworkgroupBarrier();\n".into(),
        SyncScope::Unit => "".into(),
    }
});

// WGSL has no plane barrier: a plane's is the workgroup's, here with the storage half.
wgsl_op!(SyncPlaneStorageOp, |_, _| {
    "storageBarrier();\nworkgroupBarrier();\n".into()
});
