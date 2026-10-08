//! NVPTX cube barriers, as Hopper's `mbarrier` objects in shared memory.
//!
//! A barrier is one 64-bit word. Arriving returns the phase it arrived in as a token, and waiting
//! spins on `mbarrier.try_wait` until that phase completes. A phase completes once both its
//! arrivals and the transaction bytes it expects, written by TMA loads, are all in.
//!
//! Only cube barriers are lowered: a unit barrier backs `memcpy_async`, which this backend does
//! not advertise.

use super::matrix::{as_shared, call_void};
use crate::{prelude::*, shared::plane::call_intrinsic};
use cubecl_core::ir::dialect::barrier::{
    ArriveAndExpectTxOp, ArriveAndWaitOp, ArriveOp, ExpectTxOp, InitOp, WaitOp, WaitParityOp,
};
use pliron::location::Location;

const INIT: &str = "llvm.nvvm.mbarrier.init.shared";
const ARRIVE: &str = "llvm.nvvm.mbarrier.arrive.shared";
const ARRIVE_COUNT: &str = "llvm.nvvm.mbarrier.arrive.scope.cta.space.cta";
const EXPECT_TX: &str = "llvm.nvvm.mbarrier.expect.tx.scope.cta.space.cta";

#[derive(Debug, Error)]
#[error(
    "a unit barrier backs `memcpy_async`, which the NVPTX backend does not lower; only a cube \
     barrier in shared memory is an `mbarrier`"
)]
pub struct UnitBarrierUnsupported;

/// The shared memory address of a cube barrier.
pub(crate) fn barrier_ptr(
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    barrier: Value,
    loc: Location,
) -> Result<Value> {
    // A unit barrier is a placeholder rather than memory, see `DeclareVariableOp`.
    if !barrier.get_type(ctx).deref(ctx).is::<LlvmPointerType>() {
        return input_err!(loc, UnitBarrierUnsupported);
    }
    Ok(as_shared(ctx, rw, barrier))
}

fn u32_operand(ctx: &mut Context, rw: &mut DialectConversionRewriter, value: Value) -> Value {
    let i32_ty = i32_ty(ctx);
    if value.get_type(ctx) == i32_ty {
        return value;
    }
    let op = llvm::ZExtOp::new_with_nneg(ctx, value, i32_ty, false);
    insert(ctx, rw, &op)
}

fn i64_ty(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 64, Signedness::Signless).into()
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
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
    let count = u32_operand(ctx, rw, op.arrival_count(ctx));
    call_void(ctx, rw, INIT, vec![barrier, count]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_op(
    op: &ArriveOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
    let token = arrive(ctx, rw, barrier);
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
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
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
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
    let bytes = u32_operand(ctx, rw, op.transaction_count_update(ctx));
    call_void(ctx, rw, EXPECT_TX, vec![barrier, bytes]);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait(
    op: &WaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
    let token = op.token(ctx);
    wait_token(ctx, rw, barrier, token);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn wait_parity(
    op: &WaitParityOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
    let phase = u32_operand(ctx, rw, op.phase(ctx));
    spin(
        ctx,
        rw,
        barrier,
        "mbarrier.try_wait.parity.shared::cta.b64",
        phase,
        "r",
    );
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}

pub(crate) fn arrive_and_wait(
    op: &ArriveAndWaitOp,
    ctx: &mut Context,
    rw: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let barrier = barrier_ptr(ctx, rw, op.barrier(ctx), op.loc(ctx))?;
    let token = arrive(ctx, rw, barrier);
    wait_token(ctx, rw, barrier, token);
    rw.erase_operation(ctx, op.get_operation());
    Ok(())
}
