//! NVPTX tensor memory accelerator: the bulk tensor copies between global and shared memory, and
//! the bulk async-groups that track stores.
//!
//! A load completes on an `mbarrier` as a transaction of the bytes it wrote. A store is tracked
//! by the bulk async-group it is committed in.

use crate::{
    nvptx::{address::NvptxSpace, barrier::Barrier},
    prelude::*,
};
use cubecl_core::ir::dialect::tma::{
    CommitGroupOp, TmaLoadIm2colOp, TmaLoadOp, TmaStoreOp, WaitGroupReadOp,
};
use pliron::location::Location;

const COMMIT_GROUP: &str = "llvm.nvvm.cp.async.bulk.commit.group";
const WAIT_GROUP_READ: &str = "llvm.nvvm.cp.async.bulk.wait.group.read";

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
    match resize_int(ctx, rw, value, width, Extension::Sign) {
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
    info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let rank = op.rank(ctx);
    let destination = NvptxSpace::SharedCluster.cast(ctx, rw, op.destination(ctx));
    let barrier = Barrier::new(ctx, rw, info, op.barrier(ctx)).mbarrier(loc.clone())?;
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
    info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let rank = op.rank(ctx);
    let destination = NvptxSpace::SharedCluster.cast(ctx, rw, op.destination(ctx));
    let barrier = Barrier::new(ctx, rw, info, op.barrier(ctx)).mbarrier(loc.clone())?;
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
    let source = NvptxSpace::Shared.cast(ctx, rw, op.source(ctx));
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
