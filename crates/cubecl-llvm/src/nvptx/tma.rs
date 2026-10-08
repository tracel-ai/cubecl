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

/// What a TMA copy addresses its tile with, each an integer of its own width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoordinateKind {
    /// A coordinate in the tensor, which may be negative.
    Tensor,
    /// An offset of an im2col load into the kernel window.
    Im2colOffset,
}

impl CoordinateKind {
    fn width(self) -> u32 {
        match self {
            CoordinateKind::Tensor => 32,
            CoordinateKind::Im2colOffset => 16,
        }
    }
}

impl core::fmt::Display for CoordinateKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoordinateKind::Tensor => f.write_str("coordinate"),
            CoordinateKind::Im2colOffset => f.write_str("im2col offset"),
        }
    }
}

#[derive(Debug, Error)]
#[error("a TMA {kind} is an integer, not {ty}")]
pub struct TmaOperandType {
    kind: CoordinateKind,
    ty: String,
}

/// The cube's coordinates are outermost first; PTX takes the innermost first. Each is sign
/// extended or truncated to the width of its kind.
fn coordinates(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    values: Vec<Value>,
    kind: CoordinateKind,
    loc: Location,
) -> Result<Vec<Value>> {
    values
        .into_iter()
        .rev()
        .map(
            |value| match resize_int(ctx, rw, value, kind.width(), Extension::Sign) {
                Some(value) => Ok(value),
                None => {
                    let ty = value.get_type(ctx).disp(ctx).to_string();
                    input_err!(loc.clone(), TmaOperandType { kind, ty })
                }
            },
        )
        .collect()
}

/// The optional operands a bulk copy into shared memory ends with: no multicast mask and no
/// cache hint, then the two flags that say so.
pub(crate) fn no_multicast_or_cache_hint(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
) -> Vec<Value> {
    vec![
        insert_int_const(ctx, rw, 16, 0),
        insert_int_const(ctx, rw, 64, 0),
        insert_bool_const(ctx, rw, false),
        insert_bool_const(ctx, rw, false),
    ]
}

/// The trailing operands of a tensor load: no hints, then a `cta_group` of 0, which Hopper
/// requires.
fn load_trailer(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Vec<Value> {
    let mut trailer = no_multicast_or_cache_hint(ctx, rw);
    trailer.push(insert_i32_const(ctx, rw, 0));
    trailer
}

/// The trailing operands of a tensor store: no cache hint, and the flag that says so.
fn store_trailer(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Vec<Value> {
    vec![
        insert_int_const(ctx, rw, 64, 0),
        insert_bool_const(ctx, rw, false),
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
    let coords = coordinates(ctx, rw, indices, CoordinateKind::Tensor, loc)?;

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
    let coords = coordinates(ctx, rw, indices, CoordinateKind::Tensor, loc.clone())?;
    let offsets = coordinates(ctx, rw, offsets, CoordinateKind::Im2colOffset, loc)?;

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
    let coords = coordinates(ctx, rw, indices, CoordinateKind::Tensor, loc)?;

    let mut args = vec![source, tensor_map];
    args.extend(coords);
    args.extend(store_trailer(ctx, rw));
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
