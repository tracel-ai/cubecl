use crate::prelude::*;

/// The expand half of [`IntoIterator`](core::iter::IntoIterator), which is what `zip` accepts.
pub trait IntoIteratorExpand {
    /// The iterator this turns into.
    type IntoIter: IndexedIteratorExpand;

    /// See [`IntoIterator::into_iter`](core::iter::IntoIterator::into_iter).
    fn __expand_into_iter_method(self, scope: &Scope) -> Self::IntoIter;
}

impl<I: IndexedIteratorExpand> IntoIteratorExpand for I {
    type IntoIter = I;

    fn __expand_into_iter_method(self, _scope: &Scope) -> I {
        self
    }
}
