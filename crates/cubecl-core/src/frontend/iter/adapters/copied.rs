use crate::prelude::*;

/// The expand half of [`Copied`](core::iter::Copied).
#[derive(new)]
pub struct CopiedExpand<I> {
    iter: I,
}

impl<'a, I, T> Iterable for CopiedExpand<I>
where
    I: Iterable<Item = &'a T>,
    T: DerefExpand<Target = T> + 'a,
{
    type Item = T;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, T)) {
        self.iter.expand(scope, &mut |scope, item| {
            let value = item.__expand_deref_method(scope);
            body(scope, value);
        });
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, T)) {
        self.iter.expand_unroll(scope, &mut |scope, item| {
            let value = item.__expand_deref_method(scope);
            body(scope, value);
        });
    }
}

impl<'a, I, T> IteratorExpand for CopiedExpand<I>
where
    I: IteratorExpand<Item = &'a T>,
    T: DerefExpand<Target = T> + 'a,
{
}

impl<'a, I, T> IndexedIteratorExpand for CopiedExpand<I>
where
    I: IndexedIteratorExpand<Item = &'a T>,
    T: DerefExpand<Target = T> + 'a,
{
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.iter.__expand_len_method(scope)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> T {
        self.iter.get(scope, index).__expand_deref_method(scope)
    }
}
