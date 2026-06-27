use cubecl_core::ir::dialect::tma::*;
use itertools::Itertools;

use crate::{
    cuda::cuda_op,
    shared::{CppValue, signature::op_includes},
    target::Cuda,
};

op_includes!(Cuda, [TmaStoreOp, CommitGroupOp, WaitGroupOp, WaitGroupReadOp] => "cuda/barrier");

cuda_op!(TmaLoadOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    let tensor_map = op.tensor_map(ctx).name(ctx);
    let smem_ptr = op.destination(ctx).name(ctx);
    let indices = op.indices(ctx);
    let indices = indices.iter().map(|it| it.name(ctx)).rev().join(", ");
    let rank = op.rank(ctx);
    format!(
        "cuda::device::experimental::cp_async_bulk_tensor_{rank}d_global_to_shared({smem_ptr}, &{tensor_map}, {indices}, *{barrier});"
    )
});

cuda_op!(TmaStoreOp, |op, ctx| {
    let tensor_map = op.tensor_map(ctx).name(ctx);
    let smem_ptr = op.source(ctx).name(ctx);
    let indices = op.indices(ctx);
    let indices = indices.iter().map(|it| it.name(ctx)).rev().join(", ");
    let rank = op.rank(ctx);
    format!(
        "cuda::device::experimental::cp_async_bulk_tensor_{rank}d_shared_to_global(&{tensor_map}, {indices}, {smem_ptr});"
    )
});

cuda_op!(CommitGroupOp, |_, _| {
    "cuda::device::experimental::cp_async_bulk_commit_group();".into()
});
cuda_op!(WaitGroupOp, |op, ctx| {
    let max_pending = op.max_pending(ctx).0;
    format!("cuda::device::experimental::cp_async_bulk_wait_group<{max_pending}>();")
});
cuda_op!(WaitGroupReadOp, |op, ctx| {
    let max_pending = op.max_pending(ctx).0;
    format!("cuda::device::experimental::cp_async_bulk_wait_group_read<{max_pending}>();")
});
