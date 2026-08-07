use cubecl_core::ir::{
    dialect::{barrier::*, base::ptr_value_ty},
    interfaces::TypedExt,
};

use crate::{
    cuda::{cuda_op, cuda_op_with_out},
    shared::CppValue,
};

cuda_op!(InitOp, |op, ctx| format!(
    "init({}, {});",
    op.barrier(ctx).name(ctx),
    op.arrival_count(ctx).name(ctx)
));

cuda_op!(MemCopyAsyncOp, |op, ctx| {
    let source = op.source(ctx).name(ctx);
    let barrier = op.barrier(ctx).name(ctx);
    let destination = op.destination(ctx).name(ctx);
    let source_length = op.source_length(ctx).name(ctx);
    let size = ptr_value_ty(ctx, &op.source(ctx)).size(ctx);
    match op.cooperative(ctx).0 {
        false => format!(
            "cuda::memcpy_async({destination}, {source}, {source_length} * {size}, *{barrier});"
        ),
        true => format!(
            "cuda::memcpy_async(cooperative_groups::this_thread_block(), {destination}, {source}, {source_length} * {size}, *{barrier});"
        ),
    }
});

cuda_op!(MemCopyAsyncTxOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    let source = op.source(ctx).name(ctx);
    let destination = op.destination(ctx).name(ctx);
    let source_length = op.source_length(ctx).name(ctx);
    let size = ptr_value_ty(ctx, &op.source(ctx)).size(ctx);
    format!(
        "cuda::device::memcpy_async_tx({destination}, {source}, {source_length} * {size}, *{barrier});"
    )
});

cuda_op_with_out!(ArriveOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    format!("{barrier}->arrive()")
});

cuda_op_with_out!(ArriveAndExpectTxOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    let arrive_count_update = op.arrive_count_update(ctx).name(ctx);
    let transaction_count_update = op.transaction_count_update(ctx).name(ctx);
    format!(
        "cuda::device::barrier_arrive_tx(*{barrier}, {arrive_count_update}, {transaction_count_update})"
    )
});

cuda_op!(ExpectTxOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    let transaction_count_update = op.transaction_count_update(ctx).name(ctx);
    format!("cuda::device::barrier_expect_tx(*{barrier}, {transaction_count_update});")
});

cuda_op!(WaitOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    let token = op.token(ctx).name(ctx);
    format!("{barrier}->wait(std::move({token}));")
});

cuda_op!(WaitParityOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    let phase = op.phase(ctx).name(ctx);
    format!("{barrier}->wait_parity({phase});")
});

cuda_op!(ArriveAndWaitOp, |op, ctx| {
    let barrier = op.barrier(ctx).name(ctx);
    format!("{barrier}->arrive_and_wait();")
});
