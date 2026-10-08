//! NVPTX tensor memory accelerator: the bulk tensor copies between global and shared memory, and
//! the bulk async-groups that track stores.
//!
//! A load completes on an `mbarrier` as a transaction of the bytes it wrote. A store is tracked
//! by the bulk async-group it is committed in.

use super::matrix::{as_shared, call_void};
use crate::prelude::*;
use cubecl_core::ir::{
    dialect::{
        barrier::MemCopyAsyncTxOp,
        tma::{
            CommitGroupOp, TmaLoadIm2colOp, TmaLoadOp, TmaStoreOp, WaitGroupOp, WaitGroupReadOp,
        },
    },
    interfaces::SizedType,
};
use pliron::location::Location;

const COMMIT_GROUP: &str = "llvm.nvvm.cp.async.bulk.commit.group";
const WAIT_GROUP: &str = "llvm.nvvm.cp.async.bulk.wait.group";
const WAIT_GROUP_READ: &str = "llvm.nvvm.cp.async.bulk.wait.group.read";
const BULK_GLOBAL_TO_SHARED: &str = "llvm.nvvm.cp.async.bulk.global.to.shared.cluster";

const GLOBAL_ADDRESS_SPACE: u32 = 1;
/// The loads write `.shared::cluster`, of which the cube's own shared memory is a part.
const SHARED_CLUSTER_ADDRESS_SPACE: u32 = 7;

#[derive(Debug, Error)]
#[error("a TMA {0} is an integer, not {1}")]
pub struct TmaOperandType(&'static str, String);

/// `value`, an integer, sign extended or truncated to `width` bits. Coordinates may be negative.
fn to_width(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    value: Value,
    width: u32,
    what: &'static str,
    loc: Location,
) -> Result<Value> {
    match resize_int(ctx, rw, value, width, true) {
        Some(value) => Ok(value),
        None => {
            let ty = value.get_type(ctx).disp(ctx).to_string();
            input_err!(loc, TmaOperandType(what, ty))
        }
    }
}

/// The cube's coordinates are outermost first; PTX takes the innermost first.
fn coordinates(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    values: Vec<Value>,
    width: u32,
    what: &'static str,
    loc: Location,
) -> Result<Vec<Value>> {
    values
        .into_iter()
        .rev()
        .map(|value| to_width(ctx, rw, value, width, what, loc.clone()))
        .collect()
}

fn as_shared_cluster(ctx: &mut Context, rw: &mut DialectConversionRewriter, ptr: Value) -> Value {
    let shared = as_shared(ctx, rw, ptr);
    let ty: TypeHandle = LlvmPointerType::get(ctx, SHARED_CLUSTER_ADDRESS_SPACE).into();
    let op = llvm::AddrSpaceCastOp::new(ctx, shared, ty);
    insert(ctx, rw, &op)
}

fn as_global(ctx: &mut Context, rw: &mut DialectConversionRewriter, ptr: Value) -> Value {
    let ty: TypeHandle = LlvmPointerType::get(ctx, GLOBAL_ADDRESS_SPACE).into();
    if ptr.get_type(ctx) == ty {
        return ptr;
    }
    let op = llvm::AddrSpaceCastOp::new(ctx, ptr, ty);
    insert(ctx, rw, &op)
}

/// The trailing arguments of a load: no multicast mask, no cache hint, and the flags that say
/// so, then a `cta_group` of 0, which Hopper requires.
fn load_trailer(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Vec<Value> {
    vec![
        insert_int_const(ctx, rw, 16, 0),
        insert_int_const(ctx, rw, 64, 0),
        insert_bool_const(ctx, rw, false),
        insert_bool_const(ctx, rw, false),
        insert_i32_const(ctx, rw, 0),
    ]
}

pub(crate) fn load(
    op: &TmaLoadOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let rank = op.rank(ctx);
    let destination = as_shared_cluster(ctx, rw, op.destination(ctx));
    let barrier = super::barrier::barrier_ptr(ctx, rw, op.barrier(ctx), loc.clone())?;
    let tensor_map = op.tensor_map(ctx);
    let indices = op.indices(ctx);
    let coords = coordinates(ctx, rw, indices, 32, "coordinate", loc)?;

    let mut args = vec![destination, barrier, tensor_map];
    args.extend(coords);
    args.extend(load_trailer(ctx, rw));
    let name = format!("llvm.nvvm.cp.async.bulk.tensor.g2s.tile.{rank}d");
    call_void(ctx, rw, &name, args);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn load_im2col(
    op: &TmaLoadIm2colOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let rank = op.rank(ctx);
    let destination = as_shared_cluster(ctx, rw, op.destination(ctx));
    let barrier = super::barrier::barrier_ptr(ctx, rw, op.barrier(ctx), loc.clone())?;
    let tensor_map = op.tensor_map(ctx);
    let indices = op.indices(ctx);
    let offsets = op.offsets(ctx);
    let coords = coordinates(ctx, rw, indices, 32, "coordinate", loc.clone())?;
    let offsets = coordinates(ctx, rw, offsets, 16, "im2col offset", loc)?;

    let mut args = vec![destination, barrier, tensor_map];
    args.extend(coords);
    args.extend(offsets);
    args.extend(load_trailer(ctx, rw));
    let name = format!("llvm.nvvm.cp.async.bulk.tensor.g2s.im2col.{rank}d");
    call_void(ctx, rw, &name, args);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn store(
    op: &TmaStoreOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let rank = op.rank(ctx);
    let source = as_shared(ctx, rw, op.source(ctx));
    let tensor_map = op.tensor_map(ctx);
    let indices = op.indices(ctx);
    let coords = coordinates(ctx, rw, indices, 32, "coordinate", loc)?;

    let mut args = vec![source, tensor_map];
    args.extend(coords);
    // No cache hint, and the flag that says so.
    args.push(insert_int_const(ctx, rw, 64, 0));
    args.push(insert_bool_const(ctx, rw, false));
    let name = format!("llvm.nvvm.cp.async.bulk.tensor.s2g.tile.{rank}d");
    call_void(ctx, rw, &name, args);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// The size of what `ptr` points at, read off the cube pointer type it was converted from.
pub(super) fn pointee_size(ctx: &Context, info: &OperandsInfo, ptr: Value) -> Option<usize> {
    info.lookup_operand_history(ptr)
        .into_iter()
        .rev()
        .chain(core::iter::once(ptr.get_type(ctx)))
        .find_map(|ty| {
            let ty = ty.deref(ctx);
            let ptr = ty.downcast_ref::<CubePointerType>()?;
            let inner = ptr.inner;
            let size = type_cast::<dyn SizedType>(&*inner.deref(ctx))?.size(ctx);
            Some(size)
        })
}

/// A one-dimensional bulk copy of `source_length` elements from global to shared memory, which
/// completes on the barrier as a transaction of the bytes it wrote.
pub(crate) fn memcpy_async_tx(
    op: &MemCopyAsyncTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let elem_size = pointee_size(ctx, operands_info, op.source(ctx))
        .expect("a bulk copy source points at sized elements");
    let destination = as_shared_cluster(ctx, rw, op.destination(ctx));
    let barrier = super::barrier::barrier_ptr(ctx, rw, op.barrier(ctx), loc.clone())?;
    let source = as_global(ctx, rw, op.source(ctx));
    let length = to_width(ctx, rw, op.source_length(ctx), 32, "length", loc)?;
    let elem_size = insert_i32_const(ctx, rw, elem_size as i32);
    let bytes = llvm::MulOp::new_with_overflow_flag(
        ctx,
        length,
        elem_size,
        IntegerOverflowFlagsAttr::default(),
    );
    let bytes = insert(ctx, rw, &bytes);

    let args = vec![
        destination,
        barrier,
        source,
        bytes,
        insert_int_const(ctx, rw, 16, 0),
        insert_int_const(ctx, rw, 64, 0),
        insert_bool_const(ctx, rw, false),
        insert_bool_const(ctx, rw, false),
    ];
    call_void(ctx, rw, BULK_GLOBAL_TO_SHARED, args);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn commit_group(
    op: &CommitGroupOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    call_void(ctx, rw, COMMIT_GROUP, vec![]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_group(
    op: &WaitGroupOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let max_pending = op.max_pending(ctx).0 as i32;
    let max_pending = insert_i32_const(ctx, rw, max_pending);
    call_void(ctx, rw, WAIT_GROUP, vec![max_pending]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_group_read(
    op: &WaitGroupReadOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let max_pending = op.max_pending(ctx).0 as i32;
    let max_pending = insert_i32_const(ctx, rw, max_pending);
    call_void(ctx, rw, WAIT_GROUP_READ, vec![max_pending]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}
