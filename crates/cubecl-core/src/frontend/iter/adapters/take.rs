use crate::frontend::iter::impl_iterable_indexed;
use crate::prelude::*;

/// The expand half of [`Take`](core::iter::Take).
#[derive(new)]
pub struct TakeExpand<I> {
    iter: I,
    n: NativeExpand<usize>,
}

impl_iterable_indexed!([I: IndexedIteratorExpand] TakeExpand<I>, I::Item);

impl<I: IndexedIteratorExpand> IndexedIteratorExpand for TakeExpand<I> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.iter
            .__expand_len_method(scope)
            .__expand_min_method(scope, self.n)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        self.iter.get(scope, index)
    }
}
