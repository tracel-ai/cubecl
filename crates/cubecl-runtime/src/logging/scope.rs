use core::marker::PhantomData;

use super::{Communication, ExecutionId, ExecutionKind, observer::issue};

/// Makes one communication the parent of every execution issued on this thread until it drops.
///
/// For a communication made of other executions, such as a collective emulated by transfers and
/// kernels on the calling thread: each of them names it as its parent, so an observer can charge
/// their cost to it. Scopes nest, and dropping one puts back the parent it replaced.
///
/// Inert when nothing observes: the communication is never built, and nothing is issued. Without
/// `std` there is no thread-local to hold the parent, so a scope issues its communication and
/// parents nothing: attribution is missing there rather than crossing threads.
#[must_use = "a scope parents nothing once it is dropped"]
pub struct ExecutionScope {
    replaced: Option<Option<ExecutionId>>,
    // The parent is thread-local, so a scope dropped on another thread would restore the wrong one.
    _on_this_thread: PhantomData<*const ()>,
}

impl ExecutionScope {
    /// Issues the communication `communication` builds, and opens a scope under it.
    pub fn communication(communication: impl FnOnce() -> Communication) -> Self {
        let replaced = issue(|| ExecutionKind::Communication(communication()))
            .map(|execution| parent::replace(Some(execution.id)));
        Self {
            replaced,
            _on_this_thread: PhantomData,
        }
    }

    /// The parent an execution issued on this thread now takes.
    pub(crate) fn current() -> Option<ExecutionId> {
        parent::get()
    }
}

impl Drop for ExecutionScope {
    fn drop(&mut self) {
        if let Some(replaced) = self.replaced.take() {
            parent::replace(replaced);
        }
    }
}

impl core::fmt::Debug for ExecutionScope {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExecutionScope")
            .field("opened", &self.replaced.is_some())
            .finish()
    }
}

#[cfg(feature = "std")]
mod parent {
    use super::ExecutionId;
    use core::cell::Cell;

    std::thread_local! {
        static PARENT: Cell<Option<ExecutionId>> = const { Cell::new(None) };
    }

    pub(super) fn replace(parent: Option<ExecutionId>) -> Option<ExecutionId> {
        PARENT.with(|current| current.replace(parent))
    }

    pub(super) fn get() -> Option<ExecutionId> {
        PARENT.with(Cell::get)
    }
}

#[cfg(not(feature = "std"))]
mod parent {
    use super::ExecutionId;

    pub(super) fn replace(_parent: Option<ExecutionId>) -> Option<ExecutionId> {
        None
    }

    pub(super) fn get() -> Option<ExecutionId> {
        None
    }
}
