//! Allocation modes set by a thread that has not run anything on the device yet.
//!
//! A mode belongs to a stream, and a thread's stream is only created by its first
//! operation. A mode set before then has to be kept until the stream exists, and apply
//! to that first operation.

use crate::MemoryScope;
use cubecl_runtime::{client::Client, runtime::Runtime};

const SIZE: usize = 1024;

/// A thread whose first allocation happens inside a persistent window allocates in the
/// persistent pool, and allocates outside of it again once the window closes.
pub fn a_new_threads_first_allocation_honors_its_allocation_mode<R: Runtime>(client: Client) {
    std::thread::spawn(move || {
        let persistent = client.memory_persistent_allocation((), |_| client.empty(SIZE));
        let after_window = client.empty(SIZE);

        let report = client.memory_report(MemoryScope::CurrentStream);
        let stream = report
            .streams
            .first()
            .expect("the thread's stream exists once it allocated");
        assert_eq!(
            stream.pools.persistent.usage.number_allocs, 1,
            "the allocation made inside the window, and only that one, is persistent"
        );
        assert!(
            stream.pools.persistent.usage.bytes_in_use >= SIZE as u64,
            "the persistent allocation holds its bytes"
        );
        drop((persistent, after_window));
    })
    .join()
    .expect("the allocation-mode test thread panicked");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_allocation_mode {
    () => {
        mod allocation_mode {
            use super::*;

            #[cfg(not(target_family = "wasm"))]
            #[$crate::runtime_tests::test_log::test]
            fn a_new_threads_first_allocation_honors_its_allocation_mode() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::allocation_mode::a_new_threads_first_allocation_honors_its_allocation_mode::<
                    TestRuntime,
                >(client);
            }
        }
    };
}
