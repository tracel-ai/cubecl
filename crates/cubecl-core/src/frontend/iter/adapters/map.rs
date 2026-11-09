use crate::prelude::*;

/// The expand half of [`Map`](core::iter::Map).
#[derive(new)]
pub struct MapExpand<I, F> {
    iter: I,
    f: F,
}

impl<I, F, B> Iterable for MapExpand<I, F>
where
    I: Iterable,
    F: FnMut(&Scope, I::Item) -> B,
{
    type Item = B;

    fn expand(mut self, scope: &Scope, body: &mut dyn FnMut(&Scope, B)) {
        self.iter.expand(scope, &mut |scope, item| {
            let mapped = (self.f)(scope, item);
            body(scope, mapped);
        });
    }

    fn expand_unroll(mut self, scope: &Scope, body: &mut dyn FnMut(&Scope, B)) {
        self.iter.expand_unroll(scope, &mut |scope, item| {
            let mapped = (self.f)(scope, item);
            body(scope, mapped);
        });
    }
}

impl<I, F, B> IteratorExpand for MapExpand<I, F>
where
    I: IteratorExpand,
    F: FnMut(&Scope, I::Item) -> B,
{
}

impl<I, F, B> IndexedIteratorExpand for MapExpand<I, F>
where
    I: IndexedIteratorExpand,
    F: FnMut(&Scope, I::Item) -> B,
{
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.iter.__expand_len_method(scope)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> B {
        let item = self.iter.get(scope, index);
        (self.f)(scope, item)
    }
}
