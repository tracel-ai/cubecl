use crate::frontend::iter::impl_iterable_indexed;
use crate::prelude::*;

/// The expand half of [`Rev`](core::iter::Rev).
#[derive(new)]
pub struct RevExpand<I> {
    iter: I,
}

impl_iterable_indexed!([I: IndexedIteratorExpand] RevExpand<I>, I::Item);

impl<I: IndexedIteratorExpand> IndexedIteratorExpand for RevExpand<I> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.iter.__expand_len_method(scope)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        // `len - index` is at least one, so this can't underflow.
        let index = self
            .iter
            .__expand_len_method(scope)
            .__expand_sub_method(scope, index)
            .__expand_sub_method(scope, NativeExpand::from_lit(scope, 1));
        self.iter.get(scope, index)
    }
}
