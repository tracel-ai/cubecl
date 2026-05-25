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
            let zero_i32 = const_i32(context, entry, 0)?;
            let one_i32 = const_i32(context, entry, 1)?;
            let barrier_counter_index =
                const_index(context, entry, SYNC_BARRIER_COUNTER_INDEX as i64)?;
            let stopped_counter_index =
                const_index(context, entry, SYNC_STOPPED_COUNTER_INDEX as i64)?;
            let barrier_target_index =
                const_index(context, entry, SYNC_BARRIER_TARGET_INDEX as i64)?;
            let barrier_target_value =
                atomic_load_i32(context, entry, sync_cube_state, barrier_target_index)?;
            let skip_sync = entry.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Sle,
                barrier_target_value,
                one_i32,
                location,
            ))?;
            entry.append_operation(cf::cond_br(
                context,
                skip_sync,
                done.deref(),
                wait_stopped.deref(),
                &[],
                &[],
                location,
            ));

            let stopped_value = atomic_load_i32(
                context,
                wait_stopped,
                sync_cube_state,
                stopped_counter_index,
            )?;
            let stopped_is_busy = wait_stopped.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ne,
                stopped_value,
                zero_i32,
                location,
            ))?;
            wait_stopped.append_operation(cf::cond_br(
                context,
                stopped_is_busy,
                wait_stopped.deref(),
                arrive.deref(),
                &[],
                &[],
                location,
            ));

            arrive.append_operation(llvm::fence(
                context,
                AtomicOrdering::Release,
                None,
                location,
            ));
            let arrived_prev = arrive.append_op_result(memref::atomic_rmw(
                context,
                AtomicRMWKind::AddI,
                one_i32,
                sync_cube_state,
                &[barrier_counter_index],
                location,
            ))?;
            let arrived = arrive.append_op_result(arith::addi(arrived_prev, one_i32, location))?;
            let arrived_needs_wait = arrive.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Slt,
                arrived,
                barrier_target_value,
                location,
            ))?;
            arrive.append_operation(cf::cond_br(
                context,
                arrived_needs_wait,
                wait_arrived.deref(),
                increment_stopped.deref(),
                &[],
                &[],
                location,
            ));

            let barrier_counter_value = atomic_load_i32(
                context,
                wait_arrived,
                sync_cube_state,
                barrier_counter_index,
            )?;
            let wait_more = wait_arrived.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Slt,
                barrier_counter_value,
                barrier_target_value,
                location,
            ))?;
            wait_arrived.append_operation(cf::cond_br(
                context,
                wait_more,
                wait_arrived.deref(),
                increment_stopped.deref(),
                &[],
                &[],
                location,
            ));

            increment_stopped.append_operation(llvm::fence(
                context,
                AtomicOrdering::Acquire,
                None,
                location,
            ));
            let stopped_prev = increment_stopped.append_op_result(memref::atomic_rmw(
                context,
                AtomicRMWKind::AddI,
                one_i32,
                sync_cube_state,
                &[stopped_counter_index],
                location,
            ))?;
            let stopped =
                increment_stopped.append_op_result(arith::addi(stopped_prev, one_i32, location))?;
            let should_reset = increment_stopped.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Eq,
                stopped,
                barrier_target_value,
                location,
            ))?;
            increment_stopped.append_operation(cf::cond_br(
                context,
                should_reset,
                reset.deref(),
                done.deref(),
                &[],
                &[],
                location,
            ));

            atomic_store_i32(
                context,
                reset,
                sync_cube_state,
                barrier_counter_index,
                zero_i32,
            )?;
            atomic_store_i32(
                context,
                reset,
                sync_cube_state,
                stopped_counter_index,
                zero_i32,
            )?;
            reset.append_operation(cf::br(done.deref(), &[], location));

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
