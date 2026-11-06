//! CPU cube synchronization.

use crate::{cpu::entrypoint::runtime_arg, prelude::*};
use cubecl_core::ir::dialect::{
    atomic::AtomicLoadOp,
    memory::LoadOp,
    plane::{AtomicUniformLoadOp, UniformLoadOp},
    synchronization::{SyncOp, SyncScope, SyncScopeAttr},
};
use pliron::dict_key;
use pliron_llvm::function_call_utils::lookup_or_insert_function;
use std::{
    ffi::CStr,
    sync::{
        Condvar, Mutex, OnceLock,
        atomic::{AtomicU32, Ordering},
    },
};

dict_key!(
    /// Kernel argument for the cube barrier state.
    ATTR_SYNC_CUBE_STATE,
    "sync_cube_state"
);

/// Reusable barrier shared by all units of a launch. Its layout is opaque to JIT code.
/// Short waits use atomics only. Oversubscribed cubes and delayed peers park their waiters.
pub struct CubeBarrier {
    arrived: AtomicU32,
    /// Even generations; the low bit records a possible parked waiter in this generation.
    generation: AtomicU32,
    mutex: Mutex<()>,
    ready: Condvar,
    spin_limit: u32,
}

const PARKED: u32 = 1;
const SPINS: u32 = 4096;

impl Default for CubeBarrier {
    fn default() -> Self {
        // Read on the launch thread, before workers pin themselves to individual CPUs.
        // This is only a performance hint: every wait has a bounded spin budget.
        static PARALLELISM: OnceLock<u32> = OnceLock::new();
        let spin_limit = *PARALLELISM.get_or_init(|| {
            std::thread::available_parallelism()
                .map(|parallelism| parallelism.get() as u32)
                .unwrap_or(1)
        });
        Self {
            arrived: AtomicU32::new(0),
            generation: AtomicU32::new(0),
            mutex: Mutex::new(()),
            ready: Condvar::new(),
            spin_limit,
        }
    }
}

impl CubeBarrier {
    fn wait(&self, units: u32) {
        let generation = self.generation.load(Ordering::Acquire) & !PARKED;
        // Each arrival acquires the preceding participant's writes. The last arrival
        // publishes them all when it releases the next generation.
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == units {
            self.arrived.store(0, Ordering::Relaxed);
            let previous = self
                .generation
                .swap(generation.wrapping_add(2), Ordering::Release);
            if previous & PARKED != 0 {
                // Synchronize with the predicate check and Condvar::wait below. A waiter
                // either observes the new generation or sleeps before this notification.
                let _guard = self.mutex.lock().unwrap();
                self.ready.notify_all();
            }
            return;
        }

        let spins = if units <= self.spin_limit { SPINS } else { 0 };
        for _ in 0..spins {
            if self.generation.load(Ordering::Acquire) & !PARKED != generation {
                return;
            }
            std::hint::spin_loop();
        }

        let mut guard = self.mutex.lock().unwrap();
        // Atomically register for notification and recheck the generation. If release
        // won the race, this may mark the next generation, causing only an extra wake.
        let mut observed = self.generation.fetch_or(PARKED, Ordering::AcqRel);
        while observed & !PARKED == generation {
            guard = self.ready.wait(guard).unwrap();
            observed = self.generation.load(Ordering::Acquire);
        }
    }
}

pub(crate) const CUBE_BARRIER_SYMBOL: &CStr = c"cubecl_cpu_barrier";

/// Parks waiting units so oversubscribed cubes leave CPUs available for their peers.
/// The host barrier publishes prior writes and separates successive barrier generations.
///
/// # Safety
/// `state` must point to the launch's live barrier, shared by exactly `units` participants.
/// Every invocation for that launch must use the same nonzero participant count.
pub(crate) unsafe extern "C" fn cube_barrier(state: *const CubeBarrier, units: u32) {
    // SAFETY: the JIT receives a pointer into the launch's reference-counted SharedData.
    let state = unsafe { &*state };
    state.wait(units);
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
        let module = scope.state().module;
        let ctx = scope.ctx_mut();
        let ptr_ty = LlvmPointerType::get(ctx, 0).into();
        let units_ty = i32_ty(ctx);
        let void_ty = VoidType::get(ctx).into();
        let barrier = lookup_or_insert_function(
            ctx,
            &mut SymbolTableCollection::new(),
            Box::new(module),
            CUBE_BARRIER_SYMBOL.to_str().unwrap().try_into().unwrap(),
            void_ty,
            vec![ptr_ty, units_ty],
            false,
        )
        .expect("cube barrier must have a consistent signature");
        let units = i32_const_op(ctx, units as i32);
        scope.register(&units);
        // The external call's unrestricted memory effects prevent accesses from being
        // moved across the barrier by either the CubeCL or LLVM optimizers.
        scope.register(&llvm::CallOp::new(
            ctx,
            CallOpCallable::Direct(barrier.get_symbol_name(ctx)),
            barrier.get_type(ctx),
            vec![state, units.get_result(ctx)],
        ));
    }
}

pub fn uses_cube_barrier(ctx: &Context, op: Ptr<Operation>) -> bool {
    let mut found = false;
    visit_all_ops_of_type::<SyncOp, _>(ctx, &mut found, op, |ctx, found, sync| {
        *found |= sync.scope(ctx).0 == SyncScope::Cube;
    });
    found
}

#[op_interface_impl]
impl LowerOp for UniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        scope.register(&SyncOp::new(
            scope.ctx_mut(),
            SyncScopeAttr::new(SyncScope::Cube),
        ));
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))]
    }
}

#[op_interface_impl]
impl LowerOp for AtomicUniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        scope.register(&SyncOp::new(
            scope.ctx_mut(),
            SyncScopeAttr::new(SyncScope::Cube),
        ));
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&AtomicLoadOp::new(scope.ctx_mut(), ptr))]
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, atomic::AtomicBool, mpsc},
        thread,
        time::{Duration, Instant},
    };

    #[test]
    fn a_delayed_peer_parks_even_when_the_cube_fits_the_cpu() {
        for spin_limit in [0, u32::MAX] {
            let barrier = Arc::new(CubeBarrier {
                spin_limit,
                ..Default::default()
            });
            let (done, completed) = mpsc::channel();
            let worker = {
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait(2);
                    done.send(()).unwrap();
                })
            };
            let deadline = Instant::now() + Duration::from_secs(10);
            while barrier.generation.load(Ordering::Acquire) & PARKED == 0 {
                assert!(
                    Instant::now() < deadline,
                    "waiting unit never registered to park"
                );
                thread::yield_now();
            }
            assert_eq!(completed.try_recv(), Err(mpsc::TryRecvError::Empty));
            barrier.wait(2);
            completed.recv_timeout(Duration::from_secs(10)).unwrap();
            worker.join().unwrap();
        }
    }

    #[test]
    fn release_without_parked_waiters_does_not_lock() {
        let barrier = Arc::new(CubeBarrier::default());
        let guard = barrier.mutex.lock().unwrap();
        let (done, completed) = mpsc::channel();
        let worker = {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait(1);
                done.send(()).unwrap();
            })
        };
        let result = completed.recv_timeout(Duration::from_secs(10));
        drop(guard);
        worker.join().unwrap();
        result.expect("the fast path must not need the parking mutex");
    }

    #[test]
    fn release_before_parking_does_not_lose_the_wakeup() {
        let barrier = Arc::new(CubeBarrier {
            spin_limit: 0,
            ..Default::default()
        });
        // Hold the lock to delay registration until after the final arrival releases.
        let guard = barrier.mutex.lock().unwrap();
        let (done, completed) = mpsc::channel();
        let waiter = {
            let barrier = barrier.clone();
            let done = done.clone();
            thread::spawn(move || {
                barrier.wait(2);
                done.send(()).unwrap();
            })
        };
        let deadline = Instant::now() + Duration::from_secs(10);
        while barrier.arrived.load(Ordering::Acquire) != 1 {
            assert!(Instant::now() < deadline, "waiting unit did not arrive");
            thread::yield_now();
        }
        let releaser = {
            let barrier = barrier.clone();
            thread::spawn(move || {
                barrier.wait(2);
                done.send(()).unwrap();
            })
        };
        let released = completed.recv_timeout(Duration::from_secs(10));
        drop(guard);
        released.expect("release must finish before the waiter registers");
        completed.recv_timeout(Duration::from_secs(10)).unwrap();
        waiter.join().unwrap();
        releaser.join().unwrap();
        // The late registration may mark the next generation; an extra wake is harmless.
        barrier.wait(1);
        assert_eq!(barrier.generation.load(Ordering::Acquire), 4);
    }

    #[test]
    fn both_wait_paths_publish_writes_across_generation_rollover() {
        const UNITS: usize = 8;
        const ROUNDS: u32 = 64;
        for spin_limit in [0, u32::MAX] {
            let initial_generation = u32::MAX - 3;
            let barrier = Arc::new(CubeBarrier {
                generation: AtomicU32::new(initial_generation),
                spin_limit,
                ..Default::default()
            });
            let values = Arc::new(std::array::from_fn::<_, UNITS, _>(|_| AtomicU32::new(0)));
            let failed = Arc::new(AtomicBool::new(false));
            let (done, completed) = mpsc::channel();
            let workers: Vec<_> = (0..UNITS)
                .map(|unit| {
                    let barrier = barrier.clone();
                    let values = values.clone();
                    let failed = failed.clone();
                    let done = done.clone();
                    thread::spawn(move || {
                        for round in 1..=ROUNDS {
                            if unit == round as usize % UNITS {
                                thread::yield_now();
                            }
                            values[unit].store(round, Ordering::Relaxed);
                            barrier.wait(UNITS as u32);
                            for value in values.iter() {
                                if value.load(Ordering::Relaxed) != round {
                                    failed.store(true, Ordering::Relaxed);
                                }
                            }
                            barrier.wait(UNITS as u32);
                        }
                        done.send(()).unwrap();
                    })
                })
                .collect();
            for _ in 0..UNITS {
                completed.recv_timeout(Duration::from_secs(10)).unwrap();
            }
            for worker in workers {
                worker.join().unwrap();
            }
            assert!(!failed.load(Ordering::Relaxed));
            assert_eq!(barrier.arrived.load(Ordering::Relaxed), 0);
            assert_eq!(
                barrier.generation.load(Ordering::Relaxed) & !PARKED,
                initial_generation.wrapping_add(4 * ROUNDS)
            );
        }
    }
}
