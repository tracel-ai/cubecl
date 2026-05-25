use std::ops::Deref;

use cubecl_core::ir::Synchronization;
use tracel_llvm::mlir_rs::{
    Error,
    dialect::{
        arith::{self, AtomicRMWKind, CmpiPredicate},
        cf, func,
        llvm::{self, LoadStoreOptions, attributes::AtomicOrdering},
        memref,
    },
    ir::{
        Block, BlockRef, Identifier, Location, Region,
        attribute::{FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType, MemRefType},
    },
};

use crate::compiler::{
    mlir_data::{
        SYNC_BARRIER_COUNTER_INDEX, SYNC_BARRIER_TARGET_INDEX, SYNC_STOPPED_COUNTER_INDEX,
    },
    visitor::prelude::*,
};

/// Builds the private `sync_cube` MLIR function used by the CPU backend.
///
/// Conceptually this emits the MLIR equivalent of the following Rust algorithm:
/// 1. Load `barrier_target`; return immediately when `barrier_target <= 1`.
/// 2. Spin while `stopped_counter != 0` so a previous barrier reset can complete.
/// 3. Release fence, increment `barrier_counter`, then wait until all participants arrived.
/// 4. Acquire fence, increment `stopped_counter`.
/// 5. Last participant resets both counters to zero.
pub fn add_sync_cube_function<'a>(
    context: &'a Context,
    module: &tracel_llvm::mlir_rs::ir::Module<'a>,
) -> Result<(), Error> {
    let i32_type = IntegerType::new(context, 32).into();
    let memref_type = MemRefType::new(i32_type, &[4], None, None).into();
    let func_type = TypeAttribute::new(FunctionType::new(context, &[memref_type], &[]).into());
    let location = Location::unknown(context);

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, "sync_cube"),
        func_type,
        {
            let region = Region::new();
            let entry = region.append_block(Block::new(&[(memref_type, location)]));
            let wait_stopped = region.append_block(Block::new(&[]));
            let arrive = region.append_block(Block::new(&[]));
            let wait_arrived = region.append_block(Block::new(&[]));
            let increment_stopped = region.append_block(Block::new(&[]));
            let reset = region.append_block(Block::new(&[]));
            let done = region.append_block(Block::new(&[]));

            let sync_cube_state: Value<'a, 'a> = entry.argument(0)?.into();
            let shared = SyncCubeShared {
                context,
                sync_cube_state,
                zero_i32: const_i32(context, entry, 0)?,
                one_i32: const_i32(context, entry, 1)?,
                barrier_counter_index: const_index(
                    context,
                    entry,
                    SYNC_BARRIER_COUNTER_INDEX as i64,
                )?,
                stopped_counter_index: const_index(
                    context,
                    entry,
                    SYNC_STOPPED_COUNTER_INDEX as i64,
                )?,
                barrier_target_value: atomic_load_i32(
                    context,
                    entry,
                    sync_cube_state,
                    const_index(context, entry, SYNC_BARRIER_TARGET_INDEX as i64)?,
                )?,
                location,
            };

            build_sync_cube_entry(entry, done, wait_stopped, &shared)?;
            build_sync_cube_wait_stopped(wait_stopped, arrive, &shared)?;
            build_sync_cube_arrive(arrive, wait_arrived, increment_stopped, &shared)?;
            build_sync_cube_wait_arrived(wait_arrived, increment_stopped, &shared)?;
            build_sync_cube_increment_stopped(increment_stopped, reset, done, &shared)?;
            build_sync_cube_reset(reset, done, &shared)?;

            done.append_operation(func::r#return(&[], location));
            region
        },
        &[(
            Identifier::new(context, "sym_visibility"),
            StringAttribute::new(context, "private").into(),
        )],
        location,
    ));
    Ok(())
}

struct SyncCubeShared<'a> {
    context: &'a Context,
    sync_cube_state: Value<'a, 'a>,
    zero_i32: Value<'a, 'a>,
    one_i32: Value<'a, 'a>,
    barrier_counter_index: Value<'a, 'a>,
    stopped_counter_index: Value<'a, 'a>,
    barrier_target_value: Value<'a, 'a>,
    location: Location<'a>,
}

/// Entry block:
/// - Compare `barrier_target <= 1`.
/// - Branch to `done` when sync is unnecessary, otherwise enter `wait_stopped`.
fn build_sync_cube_entry<'a>(
    entry: BlockRef<'a, 'a>,
    done: BlockRef<'a, 'a>,
    wait_stopped: BlockRef<'a, 'a>,
    shared: &SyncCubeShared<'a>,
) -> Result<(), Error> {
    let skip_sync = entry.append_op_result(arith::cmpi(
        shared.context,
        CmpiPredicate::Sle,
        shared.barrier_target_value,
        shared.one_i32,
        shared.location,
    ))?;
    entry.append_operation(cf::cond_br(
        shared.context,
        skip_sync,
        done.deref(),
        wait_stopped.deref(),
        &[],
        &[],
        shared.location,
    ));
    Ok(())
}

/// Wait block for prior reset completion:
/// - Spin while `stopped_counter != 0`.
/// - Continue to `arrive` once no thread is in the reset phase.
fn build_sync_cube_wait_stopped<'a>(
    wait_stopped: BlockRef<'a, 'a>,
    arrive: BlockRef<'a, 'a>,
    shared: &SyncCubeShared<'a>,
) -> Result<(), Error> {
    let stopped_value = atomic_load_i32(
        shared.context,
        wait_stopped,
        shared.sync_cube_state,
        shared.stopped_counter_index,
    )?;
    let stopped_is_busy = wait_stopped.append_op_result(arith::cmpi(
        shared.context,
        CmpiPredicate::Ne,
        stopped_value,
        shared.zero_i32,
        shared.location,
    ))?;
    wait_stopped.append_operation(cf::cond_br(
        shared.context,
        stopped_is_busy,
        wait_stopped.deref(),
        arrive.deref(),
        &[],
        &[],
        shared.location,
    ));
    Ok(())
}

/// Arrival block:
/// - Emit release fence.
/// - Atomically increment `barrier_counter`.
/// - If this participant is not the last to arrive, branch to `wait_arrived`.
/// - Otherwise continue to `increment_stopped`.
fn build_sync_cube_arrive<'a>(
    arrive: BlockRef<'a, 'a>,
    wait_arrived: BlockRef<'a, 'a>,
    increment_stopped: BlockRef<'a, 'a>,
    shared: &SyncCubeShared<'a>,
) -> Result<(), Error> {
    arrive.append_operation(llvm::fence(
        shared.context,
        AtomicOrdering::Release,
        None,
        shared.location,
    ));
    let arrived_prev = arrive.append_op_result(memref::atomic_rmw(
        shared.context,
        AtomicRMWKind::AddI,
        shared.one_i32,
        shared.sync_cube_state,
        &[shared.barrier_counter_index],
        shared.location,
    ))?;
    let arrived =
        arrive.append_op_result(arith::addi(arrived_prev, shared.one_i32, shared.location))?;
    let arrived_needs_wait = arrive.append_op_result(arith::cmpi(
        shared.context,
        CmpiPredicate::Slt,
        arrived,
        shared.barrier_target_value,
        shared.location,
    ))?;
    arrive.append_operation(cf::cond_br(
        shared.context,
        arrived_needs_wait,
        wait_arrived.deref(),
        increment_stopped.deref(),
        &[],
        &[],
        shared.location,
    ));
    Ok(())
}

/// Post-arrival wait block:
/// - Spin while `barrier_counter < barrier_target`.
/// - Once all participants have arrived, continue to `increment_stopped`.
fn build_sync_cube_wait_arrived<'a>(
    wait_arrived: BlockRef<'a, 'a>,
    increment_stopped: BlockRef<'a, 'a>,
    shared: &SyncCubeShared<'a>,
) -> Result<(), Error> {
    let barrier_counter_value = atomic_load_i32(
        shared.context,
        wait_arrived,
        shared.sync_cube_state,
        shared.barrier_counter_index,
    )?;
    let wait_more = wait_arrived.append_op_result(arith::cmpi(
        shared.context,
        CmpiPredicate::Slt,
        barrier_counter_value,
        shared.barrier_target_value,
        shared.location,
    ))?;
    wait_arrived.append_operation(cf::cond_br(
        shared.context,
        wait_more,
        wait_arrived.deref(),
        increment_stopped.deref(),
        &[],
        &[],
        shared.location,
    ));
    Ok(())
}

/// Completion accounting block:
/// - Emit acquire fence.
/// - Atomically increment `stopped_counter`.
/// - Last participant branches to `reset`, others branch to `done`.
fn build_sync_cube_increment_stopped<'a>(
    increment_stopped: BlockRef<'a, 'a>,
    reset: BlockRef<'a, 'a>,
    done: BlockRef<'a, 'a>,
    shared: &SyncCubeShared<'a>,
) -> Result<(), Error> {
    increment_stopped.append_operation(llvm::fence(
        shared.context,
        AtomicOrdering::Acquire,
        None,
        shared.location,
    ));
    let stopped_prev = increment_stopped.append_op_result(memref::atomic_rmw(
        shared.context,
        AtomicRMWKind::AddI,
        shared.one_i32,
        shared.sync_cube_state,
        &[shared.stopped_counter_index],
        shared.location,
    ))?;
    let stopped = increment_stopped.append_op_result(arith::addi(
        stopped_prev,
        shared.one_i32,
        shared.location,
    ))?;
    let should_reset = increment_stopped.append_op_result(arith::cmpi(
        shared.context,
        CmpiPredicate::Eq,
        stopped,
        shared.barrier_target_value,
        shared.location,
    ))?;
    increment_stopped.append_operation(cf::cond_br(
        shared.context,
        should_reset,
        reset.deref(),
        done.deref(),
        &[],
        &[],
        shared.location,
    ));
    Ok(())
}

/// Reset block executed by the last participant:
/// - Store `0` to `barrier_counter`.
/// - Store `0` to `stopped_counter`.
/// - Branch to `done`.
fn build_sync_cube_reset<'a>(
    reset: BlockRef<'a, 'a>,
    done: BlockRef<'a, 'a>,
    shared: &SyncCubeShared<'a>,
) -> Result<(), Error> {
    atomic_store_i32(
        shared.context,
        reset,
        shared.sync_cube_state,
        shared.barrier_counter_index,
        shared.zero_i32,
    )?;
    atomic_store_i32(
        shared.context,
        reset,
        shared.sync_cube_state,
        shared.stopped_counter_index,
        shared.zero_i32,
    )?;
    reset.append_operation(cf::br(done.deref(), &[], shared.location));
    Ok(())
}

fn const_i32<'a>(
    context: &'a Context,
    block: BlockRef<'a, 'a>,
    value: i64,
) -> Result<Value<'a, 'a>, Error> {
    let location = Location::unknown(context);
    block.append_op_result(arith::constant(
        context,
        IntegerAttribute::new(IntegerType::new(context, 32).into(), value).into(),
        location,
    ))
}

fn const_index<'a>(
    context: &'a Context,
    block: BlockRef<'a, 'a>,
    value: i64,
) -> Result<Value<'a, 'a>, Error> {
    let location = Location::unknown(context);
    block.append_op_result(arith::constant(
        context,
        IntegerAttribute::new(Type::index(context), value).into(),
        location,
    ))
}

fn memref_to_atomic_i32<'a>(
    context: &'a Context,
    block: BlockRef<'a, 'a>,
    memref: Value<'a, 'a>,
    index: Value<'a, 'a>,
) -> Result<Value<'a, 'a>, Error> {
    let location = Location::unknown(context);
    let index_ptr =
        block.append_op_result(memref::extract_aligned_pointer_as_index(memref, location))?;
    let int_ptr = block.append_op_result(arith::index_cast(
        index_ptr,
        IntegerType::new(context, 64).into(),
        location,
    ))?;
    let ptr = block.append_op_result(llvm::inttoptr(
        int_ptr,
        llvm::r#type::pointer(context, 0),
        location,
    ))?;
    block.append_op_result(llvm::get_element_ptr_dynamic(
        context,
        ptr,
        &[index],
        IntegerType::new(context, 32).into(),
        llvm::r#type::pointer(context, 0),
        location,
    ))
}

fn atomic_load_i32<'a>(
    context: &'a Context,
    block: BlockRef<'a, 'a>,
    memref: Value<'a, 'a>,
    index: Value<'a, 'a>,
) -> Result<Value<'a, 'a>, Error> {
    let location = Location::unknown(context);
    let ptr = memref_to_atomic_i32(context, block, memref, index)?;
    let options = LoadStoreOptions::new()
        .atomic(AtomicOrdering::Acquire)
        .align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            4,
        )));
    block.append_op_result(llvm::load(
        context,
        ptr,
        IntegerType::new(context, 32).into(),
        location,
        options,
    ))
}

fn atomic_store_i32<'a>(
    context: &'a Context,
    block: BlockRef<'a, 'a>,
    memref: Value<'a, 'a>,
    index: Value<'a, 'a>,
    value: Value<'a, 'a>,
) -> Result<(), Error> {
    let location = Location::unknown(context);
    let ptr = memref_to_atomic_i32(context, block, memref, index)?;
    let options = LoadStoreOptions::new()
        .atomic(AtomicOrdering::Release)
        .align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            4,
        )));
    block.append_operation(llvm::store(context, value, ptr, location, options));
    Ok(())
}

impl<'a> Visitor<'a> {
    pub fn visit_synchronization(&mut self, synchronization: &Synchronization) {
        match synchronization {
            Synchronization::SyncCube => {
                let func_name = FlatSymbolRefAttribute::new(self.context, "sync_cube");
                self.block.append_operation(func::call(
                    self.context,
                    func_name,
                    &[self
                        .args_manager
                        .sync_cube_state
                        .expect("sync_cube state arg missing")],
                    &[],
                    self.location,
                ));
            }
            Synchronization::SyncPlane => {} // NOOP plane size is 1 on CPU
            Synchronization::SyncStorage => {
                panic!("SyncStorage is not supported")
            }
            Synchronization::SyncAsyncProxyShared => {
                panic!("SyncProxyShared is not supported")
            }
        }
    }
}
