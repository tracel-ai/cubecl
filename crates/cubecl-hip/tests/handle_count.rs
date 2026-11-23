#![cfg(any(target_os = "linux", target_os = "windows"))]

//! A buffer's handle count, which `can_mut` reads, is its users' handles plus one at every
//! instant, however far the device thread has got with setting the buffer up. The one is the
//! allocation task's handle while it waits in the device thread's queue, and the pool's once the
//! task ran: the task's handle moves into the pool. A clone taken on the way would show for a
//! moment as a second holder, and a caller deciding whether to write in place would decide
//! differently from one run to the next — which a graph capture, whose recorded run must
//! allocate as its planning runs did, refuses to seal.

use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;

/// Allocations watched, each from its task waiting in the queue until well after it ran.
const ROUNDS: usize = 64;
const SIZE_BYTES: usize = 4096;

#[test]
fn an_allocation_never_counts_a_handle_of_the_device_thread() {
    let client = HipRuntime::client(&Default::default());
    for round in 0..ROUNDS {
        // Hold the device thread, so the allocation's task waits in its queue.
        let (held, on_device) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let blocker = {
            let client = client.clone();
            thread::spawn(move || {
                client
                    .exclusive(move || {
                        held.send(()).unwrap();
                        released.recv().unwrap();
                    })
                    .unwrap()
            })
        };
        on_device.recv().unwrap();

        let handle = client.empty(SIZE_BYTES);
        assert!(
            handle.can_mut(),
            "round {round}: one handle besides the task's"
        );

        let (sampling, sampled, above) = (
            AtomicBool::new(true),
            AtomicBool::new(false),
            AtomicBool::new(false),
        );
        thread::scope(|scope| {
            scope.spawn(|| {
                while sampling.load(Ordering::Acquire) {
                    if !handle.can_mut() {
                        above.store(true, Ordering::Release);
                    }
                    sampled.store(true, Ordering::Release);
                }
            });
            // The sampler reads before the device thread runs the task, and all along.
            while !sampled.load(Ordering::Acquire) {
                core::hint::spin_loop();
            }
            release.send(()).unwrap();
            blocker.join().unwrap();
            // A round trip: the allocation's task, queued before it, has run.
            client.check(core::iter::empty()).unwrap();
            sampling.store(false, Ordering::Release);
        });

        assert!(
            !above.load(Ordering::Acquire),
            "round {round}: the device thread held a handle of its own while it set the buffer up"
        );
        assert!(
            handle.can_mut(),
            "round {round}: one handle besides the pool's"
        );
    }
}
