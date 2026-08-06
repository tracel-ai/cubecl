//! Lowering of `sync.sync` to a spin barrier over the units of a cube.
//!
//! Every unit of a cube runs the kernel on its own thread (see the dispatcher scheduler), so a
//! cube barrier is a plain two-phase counter barrier over a pair of `u32` counters the host
//! allocates per launch and hands to the kernel through the [`ATTR_SYNC_CUBE_STATE`] argument.

use cubecl_core::ir::attributes::EntrypointInterface;
use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncScope};
use cubecl_core::ir::prelude::*;
use cubecl_core::ir::{Scope, dialect::base::OperationPtrExt};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::dict_key;
use pliron_llvm::ops as llvm;

use crate::compiler::entrypoint::runtime_arg;
use crate::compiler::polyfill::LowerOp;
use crate::compiler::polyfill::ordered_atomic::{
    atomic_fetch_add_acq_rel, atomic_load_acquire, atomic_store_release,
};

dict_key!(
    /// Marks the kernel argument pointing to the counters backing [`cube_barrier`].
    ATTR_SYNC_CUBE_STATE, "sync_cube_state"
);

/// Number of `u32` counters the host must allocate for a launch.
pub const SYNC_CUBE_STATE_LEN: usize = 2;

/// Counter of the units that reached the current barrier.
const ARRIVED: u32 = 0;
/// Counter of the units that left the current barrier.
const EXITED: u32 = 1;

/// Tells the core it is spinning, so that it can back off rather than hammer the counter. It is
/// lowered to whatever hint the target has, and to nothing at all when it has none, so a spin
/// loop is correct with or without it.
#[cube_op(name = "cpu.spin_loop_hint")]
#[result_ty(none)]
pub struct SpinLoopHintOp {}

/// Equivalent of [`std::hint::spin_loop`], for the spin loops below.
#[cube]
fn spin_loop() {
    intrinsic!(|scope| {
        let op = SpinLoopHintOp::new(scope.ctx_mut());
        scope.register(&op);
    })
}

/// Blocks until every unit of the cube reached this point, and makes the memory each of them
/// wrote before it visible to all the others.
///
/// Both counters start at zero and are left at zero, so the same pair serves every barrier of a
/// launch. `exited` is what separates two consecutive barriers: a unit that raced ahead to the
/// next barrier waits there until the last unit out of the previous one reset the counters.
#[cube]
fn cube_barrier(arrived: &Atomic<u32>, exited: &Atomic<u32>, #[comptime] units: u32) {
    // Wait for the previous barrier to be fully reset, otherwise the arrival below would be
    // wiped by that reset.
    while atomic_load_acquire(exited) != 0 {
        spin_loop();
    }

    // Publish everything written before the barrier, then register this unit as arrived.
    atomic_fetch_add_acq_rel(arrived, 1);

    // Wait for the whole cube, acquiring what every other unit published.
    while atomic_load_acquire(arrived) < units {
        spin_loop();
    }

    // The last unit out resets the counters for the next barrier. `arrived` is reset first: a
    // unit is only released into the next barrier once `exited` is back to zero, so no arrival
    // can be lost.
    if atomic_fetch_add_acq_rel(exited, 1) + 1 == units {
        atomic_store_release(arrived, 0);
        atomic_store_release(exited, 0);
    }
}

#[op_interface_impl]
impl LowerOp for SyncOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        // Read the scope out before lowering, holding the attribute borrowed would clash with
        // the ops built below.
        let sync_scope = self.scope(ctx).0;
        match sync_scope {
            // A plane is a single unit on the CPU, it is always in sync with itself.
            SyncScope::Plane => {}
            SyncScope::Cube => {
                let func = enclosing_func(ctx, self.get_operation());
                let units = func
                    .get_entrypoint_abi(ctx)
                    .expect("sync_cube must be lowered inside an entry point")
                    .cube_dim
                    .num_elems();

                // Same as a plane, a cube of a single unit needs no barrier at all.
                if units > 1 {
                    let state = runtime_arg(ctx, func, &ATTR_SYNC_CUBE_STATE);
                    let arrived = counter(scope, state, ARRIVED);
                    let exited = counter(scope, state, EXITED);
                    cube_barrier::expand(scope, &arrived.into(), &exited.into(), units);
                }
            }
            SyncScope::Device => {
                panic!("Device wide synchronization is not supported by the CPU runtime")
            }
            SyncScope::Unit => {}
        }
        vec![]
    }
}

/// Whether `op` contains a cube barrier, i.e. whether its units must run in parallel rather than
/// be queued behind each other.
pub fn uses_cube_barrier(ctx: &Context, op: Ptr<Operation>) -> bool {
    let mut found = false;
    visit_all_ops_of_type::<SyncOp, _>(ctx, &mut found, op, |ctx, found, sync| {
        *found |= sync.scope(ctx).0 == SyncScope::Cube;
    });
    found
}

/// The `u32` counter at `index` of the barrier state.
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

/// The function `op` is nested in.
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
