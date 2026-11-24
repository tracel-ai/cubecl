use crate::frontend::iter::impl_iterable_indexed;
use crate::prelude::*;

/// The expand half of [`Enumerate`](core::iter::Enumerate).
#[derive(new)]
pub struct EnumerateExpand<I> {
    iter: I,
}

impl_iterable_indexed!(
    [I: IndexedIteratorExpand] EnumerateExpand<I>,
    (NativeExpand<usize>, I::Item)
);

impl<I: IndexedIteratorExpand> IndexedIteratorExpand for EnumerateExpand<I> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.iter.__expand_len_method(scope)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        (index, self.iter.get(scope, index))
    }
}
