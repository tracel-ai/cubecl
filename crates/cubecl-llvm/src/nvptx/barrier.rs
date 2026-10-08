//! NVPTX barriers, and the copies that complete on them.
//!
//! A cube barrier is Hopper's `mbarrier`, one 64-bit word in shared memory. Arriving returns the
//! phase it arrived in as a token, and waiting spins on `mbarrier.try_wait` until that phase
//! completes. A phase completes once both its arrivals and the transaction bytes it expects,
//! written by TMA and bulk copies, are all in.
//!
//! `memcpy_async` copies synchronously: each unit's part of the copy is done before the unit
//! next arrives, and the release of its arrival with the acquire of the wait carries the copy to
//! the units that wait. A unit barrier only ever waits on its own unit's copies, so it has
//! nothing to wait for and its operations are dropped.

use crate::{
    nvptx::{address::NvptxSpace, inline_asm::InlineAsm},
    prelude::*,
    shared::{plane::call_intrinsic, to_llvm::ty::pointee_size},
};
use cubecl_core::ir::{
    dialect::barrier::{
        ArriveAndExpectTxOp, ArriveAndWaitOp, ArriveOp, ExpectTxOp, InitOp, MemCopyAsyncOp,
        MemCopyAsyncTxOp, WaitOp, WaitParityOp,
    },
    types::barrier::{BarrierLevel, BarrierType},
};
use pliron::location::Location;

const INIT: &str = "llvm.nvvm.mbarrier.init.shared";
const ARRIVE: &str = "llvm.nvvm.mbarrier.arrive.shared";
const ARRIVE_COUNT: &str = "llvm.nvvm.mbarrier.arrive.scope.cta.space.cta";
const EXPECT_TX: &str = "llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta";
const BULK_GLOBAL_TO_SHARED: &str = "llvm.nvvm.cp.async.bulk.global.to.shared.cluster";

#[derive(Debug, Error)]
#[error(
    "a TMA copy completes on an `mbarrier`, which only a cube barrier in shared memory is; a \
     unit barrier has none"
)]
pub struct UnitBarrierUnsupported;

#[derive(Debug, Error)]
#[error("a copy length is an integer, not {0}")]
pub struct CopyLengthType(String);

/// What a barrier operand is on NVPTX, read off the cube type it was converted from.
pub(crate) enum Barrier {
    /// An `mbarrier`, at its address in shared memory.
    Cube(Value),
    /// A unit barrier, which guards only the unit's own synchronous copies.
    Unit,
}

impl Barrier {
    pub(crate) fn new(
        ctx: &mut Context,
        rw: &mut DialectConversionRewriter,
        info: &OperandsInfo,
        barrier: Value,
    ) -> Self {
        let level = info
            .lookup_operand_history(barrier)
            .into_iter()
            .rev()
            .chain(core::iter::once(barrier.get_type(ctx)))
            .find_map(|ty| {
                let ty = ty.deref(ctx);
                let ptr = ty.downcast_ref::<CubePointerType>()?;
                let inner = ptr.inner.deref(ctx);
                inner.downcast_ref::<BarrierType>().map(|barrier| barrier.0)
            })
            .expect("a barrier operand points at a barrier");
        match level {
            BarrierLevel::Cube => Barrier::Cube(NvptxSpace::Shared.cast(ctx, rw, barrier)),
            BarrierLevel::Unit => Barrier::Unit,
        }
    }

    /// The `mbarrier` a copy that counts its bytes as transactions completes on.
    pub(crate) fn mbarrier(self, loc: Location) -> Result<Value> {
        match self {
            Barrier::Cube(barrier) => Ok(barrier),
            Barrier::Unit => input_err!(loc, UnitBarrierUnsupported),
        }
    }
}

/// How a wait names the phase it waits for.
enum Wait {
    /// The state an arrival returned.
    Token(Value),
    /// The parity of the phase.
    Parity(Value),
}

impl Wait {
    /// Spins until the phase completes. The label is scoped to the braces, so a kernel may hold
    /// any number of these, and the memory clobber keeps the reads the barrier guards after it.
    fn emit(self, ctx: &mut Context, rw: &mut DialectConversionRewriter, barrier: Value) {
        let i32_ty = i32_ty(ctx);
        let address = llvm::PtrToIntOp::new(ctx, barrier, i32_ty);
        let address = insert(ctx, rw, &address);

        let mut asm = InlineAsm::new().clobbers_memory();
        let address = asm.input(address, "r");
        let (test, phase) = match self {
            Wait::Token(token) => ("mbarrier.try_wait.shared::cta.b64", asm.input(token, "l")),
            Wait::Parity(parity) => {
                let parity = u32_operand(ctx, rw, parity);
                (
                    "mbarrier.try_wait.parity.shared::cta.b64",
                    asm.input(parity, "r"),
                )
            }
        };
        let template = format!(
            "{{\n.reg .pred done;\nwait:\n{test} done, [{address}], {phase};\n@!done bra wait;\n}}"
        );
        asm.emit(ctx, rw, &template);
    }
}

pub(crate) fn init(
    op: &InitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        let count = u32_operand(ctx, rw, op.arrival_count(ctx));
        call_void(ctx, rw, INIT, vec![barrier, count]);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_op(
    op: &ArriveOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    let token = match Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Barrier::Cube(barrier) => arrive(ctx, rw, barrier),
        Barrier::Unit => unit_token(ctx, rw),
    };
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![token]);
    Ok(())
}

/// Raises the bytes the current phase expects, then arrives `arrive_count_update` times. Two
/// instructions rather than `mbarrier.arrive.expect_tx`, because the count is a runtime value.
pub(crate) fn arrive_and_expect_tx(
    op: &ArriveAndExpectTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    let token = match Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Barrier::Cube(barrier) => {
            let count = u32_operand(ctx, rw, op.arrive_count_update(ctx));
            let bytes = u32_operand(ctx, rw, op.transaction_count_update(ctx));
            call_void(ctx, rw, EXPECT_TX, vec![barrier, bytes]);
            let i64_ty = i64_ty(ctx);
            call_intrinsic(ctx, rw, ARRIVE_COUNT, i64_ty, vec![barrier, count])
        }
        Barrier::Unit => unit_token(ctx, rw),
    };
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![token]);
    Ok(())
}

pub(crate) fn expect_tx(
    op: &ExpectTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        let bytes = u32_operand(ctx, rw, op.transaction_count_update(ctx));
        call_void(ctx, rw, EXPECT_TX, vec![barrier, bytes]);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait(
    op: &WaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Wait::Token(op.token(ctx)).emit(ctx, rw, barrier);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_parity(
    op: &WaitParityOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        Wait::Parity(op.phase(ctx)).emit(ctx, rw, barrier);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_and_wait(
    op: &ArriveAndWaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    if let Barrier::Cube(barrier) = Barrier::new(ctx, rw, info, op.barrier(ctx)) {
        let token = arrive(ctx, rw, barrier);
        Wait::Token(token).emit(ctx, rw, barrier);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// Copies `source_length` elements of `source` to `destination`, synchronously. A cooperative
/// copy was already split into each unit's share at the cube level.
pub(crate) fn memcpy_async(
    op: &MemCopyAsyncOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    debug_assert!(
        !op.cooperative(ctx).0,
        "a cooperative copy is split into each unit's share before it is lowered"
    );
    let (source, destination) = (op.source(ctx), op.destination(ctx));
    let bytes = copy_bytes(ctx, rw, info, source, op.source_length(ctx), op.loc(ctx))?;
    let name = format!(
        "llvm.memcpy.{}.{}.i32",
        llvm_mangled_ty(ctx, destination.get_type(ctx)),
        llvm_mangled_ty(ctx, source.get_type(ctx)),
    );
    let volatile = insert_bool_const(ctx, rw, false);
    call_void(ctx, rw, &name, vec![destination, source, bytes, volatile]);

    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

/// A one-dimensional bulk copy of `source_length` elements from global to shared memory, which
/// completes on the barrier as a transaction of the bytes it wrote.
pub(crate) fn memcpy_async_tx(
    op: &MemCopyAsyncTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
) -> Result<()> {
    let loc = op.loc(ctx);
    let barrier = Barrier::new(ctx, rw, info, op.barrier(ctx)).mbarrier(loc.clone())?;
    let bytes = copy_bytes(ctx, rw, info, op.source(ctx), op.source_length(ctx), loc)?;
    let destination = NvptxSpace::SharedCluster.cast(ctx, rw, op.destination(ctx));
    let source = NvptxSpace::Global.cast(ctx, rw, op.source(ctx));

    // No multicast mask, no cache hint, and the flags that say so.
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

/// A `u32` operand as the 32-bit integer the instructions take.
fn u32_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: Value) -> Value {
    resize_int(ctx, rw, value, 32, Extension::Zero).expect("barrier counts are integers")
}

fn arrive(ctx: &mut Context, rw: &mut DialectConversionRewriter, barrier: Value) -> Value {
    let i64_ty = i64_ty(ctx);
    call_intrinsic(ctx, rw, ARRIVE, i64_ty, vec![barrier])
}

/// The token of an arrival on a unit barrier, which no wait reads.
fn unit_token(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
    insert_int_const(ctx, rw, 64, 0)
}

/// The bytes `length` elements of what `source` points at take, as the 32-bit count the copy
/// instructions read.
fn copy_bytes(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    info: &OperandsInfo,
    source: Value,
    length: Value,
    loc: Location,
) -> Result<Value> {
    let elem_size =
        pointee_size(ctx, info, source).expect("a copy source points at sized elements");
    let Some(length) = resize_int(ctx, rw, length, 32, Extension::Zero) else {
        let ty = length.get_type(ctx).disp(ctx).to_string();
        return input_err!(loc, CopyLengthType(ty));
    };
    let elem_size = insert_i32_const(ctx, rw, elem_size as i32);
    let bytes = llvm::MulOp::new_with_overflow_flag(
        ctx,
        length,
        elem_size,
        IntegerOverflowFlagsAttr::default(),
    );
    Ok(insert(ctx, rw, &bytes))
}
