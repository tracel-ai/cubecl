//! Labels pointing to the code that allocates device memory, kept per thread.
//!
//! A label is pushed with [`AllocationLabel::push`] and popped when its
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
    pub fn push(label: &str) -> Self {
        Self::push_with(|| label.into())
    }

    /// [`push`](Self::push) a label built only when recording is on, for a
    /// caller whose label costs a format to build.
    pub fn push_with(label: impl FnOnce() -> alloc::string::String) -> Self {
        let pushed = is_recording();
        #[cfg(feature = "std")]
        if pushed {
            let label: Arc<str> = Arc::from(label());
            STACK.with(|stack| stack.borrow_mut().push(label));
        }
        #[cfg(not(feature = "std"))]
        let _ = label;
        Self {
            pushed,
            _thread_bound: core::marker::PhantomData,
        }
    }

    /// The labels this thread has pushed, outermost first. Empty when
    /// recording is off.
    pub fn current() -> Vec<Arc<str>> {
        #[cfg(feature = "std")]
        if is_recording() {
            return STACK.with(|stack| stack.borrow().clone());
        }
        Vec::new()
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

/// One thread's labels, carried to the thread that does work on its behalf.
///
/// A device runs a caller's work on a thread of its own, so the memory that
/// work allocates would otherwise carry that thread's labels, not the
/// caller's. Taken with [`current`](Self::current) where the work is handed
/// over, and [applied](Self::apply) around it where it runs.
pub struct AllocationLabelStack {
    /// Without `std` no label is kept, so none is ever applied.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    labels: Vec<Arc<str>>,
}

impl AllocationLabelStack {
    /// This thread's labels, outermost first; empty when recording is off.
    pub fn current() -> Self {
        Self {
            labels: AllocationLabel::current(),
        }
    }

    /// Run `work` with this thread's stack replaced by these labels, then put
    /// the thread's own back, even if `work` panics. Replaced rather than
    /// added to, so work run on the thread that handed it over keeps the
    /// labels it already had, not two copies of them. While recording is off,
    /// `work` just runs.
    pub fn apply<R>(self, work: impl FnOnce() -> R) -> R {
        #[cfg(feature = "std")]
        if is_recording() {
            let _restore = Restore(Some(STACK.with(|stack| stack.replace(self.labels))));
            return work();
        }
        work()
    }
}

/// Puts a thread's own labels back once the work run under carried ones ends.
#[cfg(feature = "std")]
struct Restore(Option<Vec<Arc<str>>>);

#[cfg(feature = "std")]
impl Drop for Restore {
    fn drop(&mut self) {
        if let Some(own) = self.0.take() {
            STACK.with(|stack| *stack.borrow_mut() = own);
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
            let _ignored = AllocationLabel::push("ignored");
            assert!(AllocationLabel::current().is_empty());
        }

        set_recording(true);
        {
            let _outer = AllocationLabel::push("blocks.3");
            {
                let _inner = AllocationLabel::push_with(|| "attention".into());
                let labels = AllocationLabel::current();
                let labels: Vec<&str> = labels.iter().map(|label| &**label).collect();
                assert_eq!(labels, ["blocks.3", "attention"]);
            }
            assert_eq!(AllocationLabel::current().len(), 1);

            // A guard pushed while off pops nothing once recording is on.
            set_recording(false);
            let off = AllocationLabel::push("off");
            set_recording(true);
            drop(off);
            assert_eq!(AllocationLabel::current().len(), 1);
        }
        assert!(AllocationLabel::current().is_empty());

        carried_labels_replace_the_running_thread_s_own_while_the_work_runs();
        set_recording(false);
    }

    /// Part of the one test above: recording is on.
    fn carried_labels_replace_the_running_thread_s_own_while_the_work_runs() {
        let caller = {
            let _model = AllocationLabel::push("model");
            let _step = AllocationLabel::push("step 3");
            AllocationLabelStack::current()
        };
        let seen = std::thread::spawn(move || {
            let _own = AllocationLabel::push("worker");
            let during = caller.apply(AllocationLabel::current);
            (during, AllocationLabel::current())
        })
        .join()
        .unwrap();

        let names = |labels: &[Arc<str>]| {
            labels
                .iter()
                .map(|l| alloc::string::String::from(&**l))
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&seen.0), ["model", "step 3"]);
        assert_eq!(names(&seen.1), ["worker"]);
    }
}
