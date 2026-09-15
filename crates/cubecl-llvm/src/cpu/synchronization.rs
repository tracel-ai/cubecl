//! CPU cube synchronization.

use crate::{
    cpu::{
        entrypoint::runtime_arg,
        ordered_atomic::{atomic_fetch_add_acq_rel, atomic_load_acquire, atomic_store_release},
    },
    prelude::*,
};
use cubecl_core::{
    ir::dialect::synchronization::{SyncOp, SyncScope},
    prelude::*,
};
use pliron::dict_key;

dict_key!(
    /// Kernel argument for the cube barrier state.
    ATTR_SYNC_CUBE_STATE, "sync_cube_state"
);

/// Number of barrier counters required per launch.
pub const SYNC_CUBE_STATE_LEN: usize = 2;

/// Units that reached the barrier.
const ARRIVED: u32 = 0;
/// Units that left the barrier.
const EXITED: u32 = 1;

#[cube_op(name = "cpu.spin_loop_hint")]
#[result_ty(none)]
pub struct SpinLoopHintOp {}

#[cube]
fn spin_loop() {
    intrinsic!(|scope| {
        let op = SpinLoopHintOp::new(scope.ctx_mut());
        scope.register(&op);
    })
}

/// Synchronizes all units and makes prior writes visible across the cube.
/// Counters must start at zero and return to zero after each barrier.
#[cube]
fn cube_barrier(arrived: &Atomic<u32>, exited: &Atomic<u32>, #[comptime] units: u32) {
    while atomic_load_acquire(exited) != 0 {
        spin_loop();
    }

    atomic_fetch_add_acq_rel(arrived, 1);

    while atomic_load_acquire(arrived) < units {
        spin_loop();
    }

    // Arrival counters must reset before units can enter the next barrier.
    if atomic_fetch_add_acq_rel(exited, 1) + 1 == units {
        atomic_store_release(arrived, 0);
        atomic_store_release(exited, 0);
    }
}

pub fn lower_sync_cube(scope: &Scope, op: Ptr<Operation>) {
    let ctx = scope.ctx();
    let func = enclosing_func(ctx, op);
    let units = func
        .get_entrypoint_abi(ctx)
        .expect("sync_cube must be lowered inside an entry point")
        .cube_dim
        .num_elems();

    if units > 1 {
        let state = runtime_arg(ctx, func, &ATTR_SYNC_CUBE_STATE);
        let arrived = counter(scope, state, ARRIVED);
        let exited = counter(scope, state, EXITED);
        cube_barrier::expand(scope, &arrived.into(), &exited.into(), units);
    }
}

pub fn uses_cube_barrier(ctx: &Context, op: Ptr<Operation>) -> bool {
    let mut found = false;
    visit_all_ops_of_type::<SyncOp, _>(ctx, &mut found, op, |ctx, found, sync| {
        *found |= sync.scope(ctx).0 == SyncScope::Cube;
    });
    found
}

fn counter(scope: &Scope, state: Value, index: u32) -> Value {
    let u32_ty = IntegerType::get(scope.ctx(), 32, Signedness::Signless).into();
    let gep = llvm::GetElementPtrOp::new(
        scope.ctx_mut(),
        state,
        vec![llvm::GepIndex::Constant(index)],
        u32_ty,
    );
    scope.register(&gep);
    gep.get_result(scope.ctx())
}

fn enclosing_func(ctx: &Context, op: Ptr<Operation>) -> FuncOp {
    let mut current = op;
    loop {
        current = current
            .deref(ctx)
            .get_parent_op(ctx)
            .expect("op must be nested in a function");
        if let Some(func) = current.as_op::<FuncOp>(ctx) {
            return func;
        }
    }
}
