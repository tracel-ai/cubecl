use crate::frontend::iter::impl_iterable_indexed;
use crate::prelude::*;

/// The expand half of [`Skip`](core::iter::Skip).
#[derive(new)]
pub struct SkipExpand<I> {
    iter: I,
    n: NativeExpand<usize>,
}

impl_iterable_indexed!([I: IndexedIteratorExpand] SkipExpand<I>, I::Item);

impl<I: IndexedIteratorExpand> IndexedIteratorExpand for SkipExpand<I> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.iter
            .__expand_len_method(scope)
            .__expand_saturating_sub_method(scope, self.n)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        let index = index.__expand_add_method(scope, self.n);
        self.iter.get(scope, index)
    }
}
