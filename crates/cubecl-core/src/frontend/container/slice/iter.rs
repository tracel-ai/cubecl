use crate::frontend::iter::impl_iterable_indexed;
use crate::prelude::*;

use super::SliceExpand;

impl<E: CubePrimitive> SliceExpand<E> {
    /// See [`slice::iter`].
    pub fn __expand_iter_method(&self, _scope: &Scope) -> IterExpand<'_, E> {
        IterExpand { slice: self }
    }

    /// See [`slice::iter_mut`].
    pub fn __expand_iter_mut_method(&mut self, _scope: &Scope) -> IterMutExpand<'_, E> {
        IterMutExpand { slice: self }
    }
}

/// The expand half of [`Iter`](core::slice::Iter).
pub struct IterExpand<'a, E: CubePrimitive> {
    slice: &'a SliceExpand<E>,
}

impl_iterable_indexed!(['a, E: CubePrimitive] IterExpand<'a, E>, &'a NativeExpand<E>);

impl<E: CubePrimitive> IndexedIteratorExpand for IterExpand<'_, E> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.slice.__expand_len_method(scope)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        let slice = self.slice;
        slice.__expand_index_method(scope, index)
    }
}

/// The expand half of [`IterMut`](core::slice::IterMut).
pub struct IterMutExpand<'a, E: CubePrimitive> {
    // Shared because `get` hands out many `&'a mut` references, each to a separate kernel value
    // rather than to anything in this slice expand.
    slice: &'a SliceExpand<E>,
}

impl_iterable_indexed!(['a, E: CubePrimitive] IterMutExpand<'a, E>, &'a mut NativeExpand<E>);

impl<E: CubePrimitive> IndexedIteratorExpand for IterMutExpand<'_, E> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.slice.__expand_len_method(scope)
    }

    fn get(&mut self, scope: &Scope, index: NativeExpand<usize>) -> Self::Item {
        let slice = self.slice;
        slice
            .__expand_as_mut_unchecked_method(scope)
            .__expand_index_mut_method(scope, index)
    }
}

impl<'a, E: CubePrimitive> IntoIteratorExpand for &'a SliceExpand<E> {
    type IntoIter = IterExpand<'a, E>;

    fn __expand_into_iter_method(self, scope: &Scope) -> Self::IntoIter {
        self.__expand_iter_method(scope)
    }
}

impl<'a, E: CubePrimitive> IntoIteratorExpand for &'a mut SliceExpand<E> {
    type IntoIter = IterMutExpand<'a, E>;

    fn __expand_into_iter_method(self, scope: &Scope) -> Self::IntoIter {
        self.__expand_iter_mut_method(scope)
    }
}
