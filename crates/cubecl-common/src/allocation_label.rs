//! Labels pointing to the code that allocates device memory, kept per thread.
//!
//! A label is pushed with [`AllocationLabel::enter`] and popped when its
//! guard drops, so labels nest: loading `blocks.3` while loading its
//! `attention` field reads as `["blocks.3", "attention"]`. The runtime reads
//! the stack ([`AllocationLabel::current`]) when a memory handle is created and
//! keeps it with the allocation, so a memory report can say who holds what part
//! of the code.
//!
//! The stack lives in this crate rather than the runtime's so that code which
//! never names a runtime — a framework's module loader, say — can label what
//! it allocates.
//!
//! Nothing is kept unless [recording](set_recording) is on. Without `std` there
//! are no threads to keep a stack for, and every label is ignored.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

static RECORDING: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "std")]
std::thread_local! {
    static STACK: core::cell::RefCell<Vec<Arc<str>>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// Turn label recording on or off for the whole process. The runtime sets it
/// from its memory tracing configuration.
pub fn set_recording(recording: bool) {
    RECORDING.store(recording, Ordering::Relaxed);
}

/// Whether labels are being recorded.
pub fn is_recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// A label on the allocations this thread makes.
#[must_use = "the label is popped as soon as the guard drops"]
pub struct AllocationLabel {
    /// Whether this guard pushed a label, so a guard made while recording was
    /// off never pops one pushed by another. Without `std` nothing is pushed.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pushed: bool,
    _thread_bound: core::marker::PhantomData<*const ()>,
}

impl AllocationLabel {
    /// Push `label` onto this thread's stack until the guard drops.
    pub fn enter(label: &str) -> Self {
        Self::push(|| Arc::from(label))
    }

    /// [`enter`](Self::enter) a label built only when recording is on, for a
    /// caller whose label costs a format to build.
    pub fn enter_with(label: impl FnOnce() -> alloc::string::String) -> Self {
        Self::push(|| Arc::from(label()))
    }

    /// The labels this thread has entered, outermost first. Empty when
    /// recording is off.
    pub fn current() -> Vec<Arc<str>> {
        #[cfg(feature = "std")]
        if is_recording() {
            return STACK.with(|stack| stack.borrow().clone());
        }
        Vec::new()
    }

    fn push(label: impl FnOnce() -> Arc<str>) -> Self {
        let pushed = is_recording();
        #[cfg(feature = "std")]
        if pushed {
            let label = label();
            STACK.with(|stack| stack.borrow_mut().push(label));
        }
        #[cfg(not(feature = "std"))]
        let _ = label;
        Self {
            pushed,
            _thread_bound: core::marker::PhantomData,
        }
    }
}

impl Drop for AllocationLabel {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        if self.pushed {
            STACK.with(|stack| stack.borrow_mut().pop());
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn labels_nest_pop_in_order_and_cost_nothing_while_off() {
        set_recording(false);
        {
            let _ignored = AllocationLabel::enter("ignored");
            assert!(AllocationLabel::current().is_empty());
        }

        set_recording(true);
        {
            let _outer = AllocationLabel::enter("blocks.3");
            {
                let _inner = AllocationLabel::enter_with(|| "attention".into());
                let labels = AllocationLabel::current();
                let labels: Vec<&str> = labels.iter().map(|label| &**label).collect();
                assert_eq!(labels, ["blocks.3", "attention"]);
            }
            assert_eq!(AllocationLabel::current().len(), 1);

            // A guard entered while off pops nothing once recording is on.
            set_recording(false);
            let off = AllocationLabel::enter("off");
            set_recording(true);
            drop(off);
            assert_eq!(AllocationLabel::current().len(), 1);
        }
        assert!(AllocationLabel::current().is_empty());
        set_recording(false);
    }
}
