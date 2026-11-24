use crate::frontend::iter::impl_iterable_indexed;
use crate::prelude::*;

/// The expand half of [`Zip`](core::iter::Zip).
#[derive(new)]
pub struct ZipExpand<A, B> {
    a: A,
    b: B,
}

impl_iterable_indexed!(
    [A: IndexedIteratorExpand, B: IndexedIteratorExpand] ZipExpand<A, B>,
    (A::Item, B::Item)
);

impl<A: IndexedIteratorExpand, B: IndexedIteratorExpand> IndexedIteratorExpand for ZipExpand<A, B> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let a = self.a.__expand_len_method(scope);
        let b = self.b.__expand_len_method(scope);
        a.__expand_min_method(scope, b)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        (self.a.get(scope, index), self.b.get(scope, index))
    }
}
