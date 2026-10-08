//! NVPTX barriers.
//!
//! A cube barrier is Hopper's `mbarrier`, one 64-bit word in shared memory. Arriving returns the
//! phase it arrived in as a token, and waiting spins on `mbarrier.try_wait` until that phase
//! completes. A phase completes once both its arrivals and the transaction bytes it expects,
//! written by TMA loads, are all in.
//!
//! `memcpy_async` copies synchronously: each unit's part of the copy is done before the unit
//! next arrives, and the release of its arrival with the acquire of the wait carries the copy to
//! the units that wait. A unit barrier only ever waits on its own unit's copies, so it has
//! nothing to wait for and its operations are dropped.

use super::{
    matrix::{as_shared, call_void},
    tma::pointee_size,
};
use crate::{prelude::*, shared::plane::call_intrinsic};
use cubecl_core::ir::dialect::barrier::{
    ArriveAndExpectTxOp, ArriveAndWaitOp, ArriveOp, ExpectTxOp, InitOp, MemCopyAsyncOp, WaitOp,
    WaitParityOp,
};
use pliron::location::Location;

const INIT: &str = "llvm.nvvm.mbarrier.init.shared";
const ARRIVE: &str = "llvm.nvvm.mbarrier.arrive.shared";
const ARRIVE_COUNT: &str = "llvm.nvvm.mbarrier.arrive.scope.cta.space.cta";
const EXPECT_TX: &str = "llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta";

const TID: [&str; 3] = [
    "llvm.nvvm.read.ptx.sreg.tid.x",
    "llvm.nvvm.read.ptx.sreg.tid.y",
    "llvm.nvvm.read.ptx.sreg.tid.z",
];
const NTID: [&str; 3] = [
    "llvm.nvvm.read.ptx.sreg.ntid.x",
    "llvm.nvvm.read.ptx.sreg.ntid.y",
    "llvm.nvvm.read.ptx.sreg.ntid.z",
];

#[derive(Debug, Error)]
#[error(
    "a TMA copy completes on an `mbarrier`, which only a cube barrier in shared memory is; a \
     unit barrier has none"
)]
pub struct UnitBarrierUnsupported;

/// The shared memory address of a cube barrier, or `None` for a unit barrier, which is a
/// placeholder rather than memory (see `DeclareVariableOp`).
fn cube_barrier(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    barrier: Value,
) -> Option<Value> {
    if !barrier.get_type(ctx).deref(ctx).is::<LlvmPointerType>() {
        return None;
    }
    Some(as_shared(ctx, rw, barrier))
}

/// The shared memory address of the cube barrier a TMA copy completes on.
pub(crate) fn barrier_ptr(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    barrier: Value,
    loc: Location,
) -> Result<Value> {
    match cube_barrier(ctx, rw, barrier) {
        Some(barrier) => Ok(barrier),
        None => input_err!(loc, UnitBarrierUnsupported),
    }
}

/// The token of an arrival on a unit barrier, which no wait reads.
fn unit_token(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
    insert_int_const(ctx, rw, 64, 0)
}

/// A `u32` operand as the 32-bit integer the instructions take.
fn u32_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: Value) -> Value {
    resize_int(ctx, rw, value, 32, false).expect("barrier counts are integers")
}

/// The 32-bit shared address an inline `mbarrier` instruction takes.
fn address(ctx: &mut Context, rw: &mut DialectConversionRewriter, barrier: Value) -> Value {
    let i32_ty = i32_ty(ctx);
    let op = llvm::PtrToIntOp::new(ctx, barrier, i32_ty);
    insert(ctx, rw, &op)
}

/// Spins until `test` completes the phase. The label is scoped to the braces, so a kernel may
/// hold any number of these. The memory clobber keeps the shared memory reads the barrier
/// guards after it.
fn spin(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    barrier: Value,
    test: &str,
    operand: Value,
    constraint: &str,
) {
    let address = address(ctx, rw, barrier);
    let template =
        format!("{{\n.reg .pred done;\nwait:\n{test} done, [$0], $1;\n@!done bra wait;\n}}");
    let constraints = format!("r,{constraint},~{{memory}}");
    let void_ty = VoidType::get(ctx).into();
    let asm = llvm::InlineAsmOp::new(
        ctx,
        void_ty,
        vec![address, operand],
        &template,
        &constraints,
        true,
    );
    rw.insert_op(ctx, &asm);
}

fn wait_token(ctx: &mut Context, rw: &mut DialectConversionRewriter, barrier: Value, token: Value) {
    spin(
        ctx,
        rw,
        barrier,
        "mbarrier.try_wait.shared::cta.b64",
        token,
        "l",
    );
}

fn arrive(ctx: &mut Context, rw: &mut DialectConversionRewriter, barrier: Value) -> Value {
    let i64_ty = i64_ty(ctx);
    call_intrinsic(ctx, rw, ARRIVE, i64_ty, vec![barrier])
}

pub(crate) fn init(
    op: &InitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if let Some(barrier) = cube_barrier(ctx, rw, op.barrier(ctx)) {
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
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let token = match cube_barrier(ctx, rw, op.barrier(ctx)) {
        Some(barrier) => arrive(ctx, rw, barrier),
        None => unit_token(ctx, rw),
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
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let Some(barrier) = cube_barrier(ctx, rw, op.barrier(ctx)) else {
        let token = unit_token(ctx, rw);
        rw.replace_operation_with_values(ctx, op.get_operation(), vec![token]);
        return Ok(());
    };
    let count = u32_operand(ctx, rw, op.arrive_count_update(ctx));
    let bytes = u32_operand(ctx, rw, op.transaction_count_update(ctx));
    call_void(ctx, rw, EXPECT_TX, vec![barrier, bytes]);
    let i64_ty = i64_ty(ctx);
    let token = call_intrinsic(ctx, rw, ARRIVE_COUNT, i64_ty, vec![barrier, count]);
    rw.replace_operation_with_values(ctx, op.get_operation(), vec![token]);
    Ok(())
}

pub(crate) fn expect_tx(
    op: &ExpectTxOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if let Some(barrier) = cube_barrier(ctx, rw, op.barrier(ctx)) {
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
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if let Some(barrier) = cube_barrier(ctx, rw, op.barrier(ctx)) {
        let token = op.token(ctx);
        wait_token(ctx, rw, barrier, token);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_parity(
    op: &WaitParityOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if let Some(barrier) = cube_barrier(ctx, rw, op.barrier(ctx)) {
        let phase = u32_operand(ctx, rw, op.phase(ctx));
        spin(
            ctx,
            rw,
            barrier,
            "mbarrier.try_wait.parity.shared::cta.b64",
            phase,
            "r",
        );
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_and_wait(
    op: &ArriveAndWaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    if let Some(barrier) = cube_barrier(ctx, rw, op.barrier(ctx)) {
        let token = arrive(ctx, rw, barrier);
        wait_token(ctx, rw, barrier, token);
    }
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

fn read_sreg(ctx: &mut Context, rw: &mut DialectConversionRewriter, name: &str) -> Value {
    let ty = i32_ty(ctx);
    call_intrinsic(ctx, rw, name, ty, vec![])
}

fn udiv(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op = llvm::UDivOp::new(ctx, lhs, rhs);
    insert(ctx, rw, &op)
}

fn sub(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::SubOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

fn mul(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

fn add(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let op =
        llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, IntegerOverflowFlagsAttr::default());
    insert(ctx, rw, &op)
}

fn umin(ctx: &mut Context, rw: &mut DialectConversionRewriter, lhs: Value, rhs: Value) -> Value {
    let ty = i32_ty(ctx);
    call_intrinsic(ctx, rw, "llvm.umin.i32", ty, vec![lhs, rhs])
}

/// The unit's position in its cube and the cube's unit count, as `UNIT_POS` and `CUBE_DIM`.
fn unit_in_cube(ctx: &mut Context, rw: &mut DialectConversionRewriter) -> (Value, Value) {
    let tid = TID.map(|name| read_sreg(ctx, rw, name));
    let ntid = NTID.map(|name| read_sreg(ctx, rw, name));
    let plane_xy = mul(ctx, rw, ntid[0], ntid[1]);
    let units = mul(ctx, rw, plane_xy, ntid[2]);
    let row = mul(ctx, rw, tid[1], ntid[0]);
    let layer = mul(ctx, rw, tid[2], plane_xy);
    let pos = add(ctx, rw, tid[0], row);
    let pos = add(ctx, rw, pos, layer);
    (pos, units)
}

/// Copies `source_length` elements of `source` to `destination`, synchronously. A cooperative
/// copy, which every unit of the cube makes, gives each unit a contiguous share of the elements.
pub(crate) fn memcpy_async(
    op: &MemCopyAsyncOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let source = op.source(ctx);
    let destination = op.destination(ctx);
    let elem_size =
        pointee_size(ctx, operands_info, source).expect("a copy source points at sized elements");
    let length =
        resize_int(ctx, rw, op.source_length(ctx), 32, false).expect("a copy length is an integer");
    let elem_size = insert_i32_const(ctx, rw, elem_size as i32);

    let (first, count) = if op.cooperative(ctx).0 {
        // `ceil(length / units)` each, the last shares clamped to the end.
        let (pos, units) = unit_in_cube(ctx, rw);
        let one = insert_i32_const(ctx, rw, 1);
        let units_less_one = sub(ctx, rw, units, one);
        let rounded_up = add(ctx, rw, length, units_less_one);
        let share = udiv(ctx, rw, rounded_up, units);
        let start = mul(ctx, rw, pos, share);
        let start = umin(ctx, rw, start, length);
        let end = add(ctx, rw, start, share);
        let end = umin(ctx, rw, end, length);
        let count = sub(ctx, rw, end, start);
        (start, count)
    } else {
        (insert_i32_const(ctx, rw, 0), length)
    };

    let offset = mul(ctx, rw, first, elem_size);
    let bytes = mul(ctx, rw, count, elem_size);
    let i8_ty = int_ty(ctx, 8);
    let gep =
        llvm::GetElementPtrOp::new(ctx, destination, vec![llvm::GepIndex::Value(offset)], i8_ty);
    let destination = insert(ctx, rw, &gep);
    let gep = llvm::GetElementPtrOp::new(ctx, source, vec![llvm::GepIndex::Value(offset)], i8_ty);
    let source = insert(ctx, rw, &gep);

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
