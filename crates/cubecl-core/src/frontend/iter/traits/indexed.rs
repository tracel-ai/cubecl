use crate::frontend::iter::{EnumerateExpand, RevExpand, SkipExpand, TakeExpand, ZipExpand};
use crate::prelude::*;

/// An iterator with a known length and random access to its items.
///
/// Slice iterators and ranges are indexed, and adapters over them stay indexed. This has no
/// counterpart in the public API of [`core::iter`], which `next` makes possible for any iterator.
pub trait IndexedIteratorExpand: IteratorExpand {
    /// See [`ExactSizeIterator::len`](core::iter::ExactSizeIterator::len).
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize>;

    /// The item at `index`, which must be less than the length.
    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item;

    /// See [`Iterator::zip`](core::iter::Iterator::zip).
    fn __expand_zip_method<U>(
        self,
        scope: &Scope,
        other: IterArg<U>,
    ) -> ZipExpand<Self, U::IntoIter>
    where
        U: IntoIteratorExpand,
    {
        ZipExpand::new(self, other.0.__expand_into_iter_method(scope))
    }

    /// See [`Iterator::enumerate`](core::iter::Iterator::enumerate).
    fn __expand_enumerate_method(self, _scope: &Scope) -> EnumerateExpand<Self> {
        EnumerateExpand::new(self)
    }

    /// See [`Iterator::rev`](core::iter::Iterator::rev).
    fn __expand_rev_method(self, _scope: &Scope) -> RevExpand<Self> {
        RevExpand::new(self)
    }

    /// See [`Iterator::take`](core::iter::Iterator::take).
    fn __expand_take_method(self, _scope: &Scope, n: NativeExpand<usize>) -> TakeExpand<Self> {
        TakeExpand::new(self, n)
    }

    /// See [`Iterator::skip`](core::iter::Iterator::skip).
    fn __expand_skip_method(self, _scope: &Scope, n: NativeExpand<usize>) -> SkipExpand<Self> {
        SkipExpand::new(self, n)
    }
}

/// Expand the loop of an indexed iterator.
///
/// This calls [`Iterable::expand`] rather than [`for_expand`], which leaves out the loop of a
/// constant length of one that `any` and `all` need to break out of.
pub(crate) fn expand_indexed<I: IndexedIteratorExpand>(
    mut iter: I,
    scope: &Scope,
    body: &mut dyn FnMut(&Scope, I::Item),
) {
    let start = NativeExpand::<usize>::from_lit(scope, 0);
    let end = iter.__expand_len_method(scope);
    RangeExpand::new(start, end).expand(scope, &mut |scope, index| {
        let item = iter.get(scope, index);
        body(scope, item);
    });
}
