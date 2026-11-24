use core::marker::PhantomData;

use super::{Communication, ExecutionId, ExecutionKind, OBSERVERS};

/// Makes one communication the parent of every execution issued on this thread until it drops.
///
/// For a communication made of other executions, such as a collective emulated by transfers and
/// kernels on the calling thread: each of them names it as its parent, so an observer can charge
/// their cost to it. Scopes nest, and dropping one puts back the parent it replaced.
///
/// Inert when nothing observes: the communication is never built, and nothing is issued. Without
/// `std` there is no thread-local to hold the parent, so a scope issues its communication and
/// parents nothing: attribution is missing there rather than crossing threads.
///
/// # Assumptions
///
/// Scopes drop in the reverse order they were opened.
#[derive(Debug)]
#[must_use = "a scope parents nothing once it is dropped"]
pub struct ExecutionScope {
    replaced: Option<Option<ExecutionId>>,
    // The parent is thread-local, so a scope dropped on another thread would restore the wrong one.
    _on_this_thread: PhantomData<*const ()>,
}

impl ExecutionScope {
    /// Issues the communication `communication` builds, and opens a scope under it.
    pub fn communication(communication: impl FnOnce() -> Communication) -> Self {
        Self::under(
            OBSERVERS
                .issue(|| ExecutionKind::Communication(communication()))
                .map(|reported| reported.id()),
        )
    }
}

impl ExecutionScope {
    /// Opens a scope under `parent`, an execution already issued, possibly on another thread.
    /// Inert when there is none, leaving whatever scope this thread has open.
    pub(crate) fn under(parent: Option<ExecutionId>) -> Self {
        Self {
            replaced: parent.map(|parent| parent::replace(Some(parent))),
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
        let Some(replaced) = self.replaced.take() else {
            return;
        };
        parent::replace(replaced);
    }
}

/// The parent of the executions this thread issues.
#[cfg(feature = "std")]
mod parent {
    use super::ExecutionId;
    use core::cell::Cell;

    std::thread_local! {
        static PARENT: Cell<Option<ExecutionId>> = const { Cell::new(None) };
    }

    pub fn replace(parent: Option<ExecutionId>) -> Option<ExecutionId> {
        PARENT.with(|current| current.replace(parent))
    }

    pub fn get() -> Option<ExecutionId> {
        PARENT.with(Cell::get)
    }
}

/// Without `std` there is no thread to be local to: nothing is parented.
#[cfg(not(feature = "std"))]
mod parent {
    use super::ExecutionId;

    pub fn replace(_parent: Option<ExecutionId>) -> Option<ExecutionId> {
        None
    }

    pub fn get() -> Option<ExecutionId> {
        None
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn a_scope_under_a_parent_parents_what_is_issued_inside_it() {
        let parent = ExecutionId::mint();
        let scope = ExecutionScope::under(Some(parent));
        assert_eq!(ExecutionScope::current(), Some(parent));
        drop(scope);
    }

    #[test]
    fn a_scope_under_no_parent_leaves_the_open_one() {
        let parent = ExecutionId::mint();
        let _outer = ExecutionScope::under(Some(parent));
        let _inert = ExecutionScope::under(None);
        assert_eq!(ExecutionScope::current(), Some(parent));
    }

    #[test]
    fn a_dropped_scope_puts_back_the_parent_it_replaced() {
        let outer = ExecutionId::mint();
        let _outer = ExecutionScope::under(Some(outer));
        drop(ExecutionScope::under(Some(ExecutionId::mint())));
        assert_eq!(ExecutionScope::current(), Some(outer));
    }
}
