use alloc::vec::Vec;
use cubecl_ir::Scope;
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use core::ops::{Deref, Index, IndexMut};

/// A sequence of [cube types](CubeType) that is inlined during compilation.
///
/// In other words, it allows you to group a dynamic amount of variables at compile time.
///
/// All methods [push](Sequence::push), [index](Sequence::index) and
/// [`into_iter`](Sequence::into_iter) are executed _during_ compilation and don't add any overhead
/// on the generated kernel.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Sequence<T: CubeType> {
    values: Vec<T>,
}

impl<T: CubeType> Default for Sequence<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: CubeType> IntoExpand for SequenceExpand<T> {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self
    }
}

impl<T: CubeType> IntoMut for Sequence<T> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}
impl<T: CubeType> CubeDebug for Sequence<T> {}

impl<T: CubeType<ExpandType: Clone>> DerefExpand for SequenceExpand<T> {
    type Target = Self;

    fn __expand_deref_method(&self, _: &Scope) -> Self::Target {
        self.clone()
    }
}

impl<T: CubeType> Sequence<T> {
    pub fn reverse(&mut self) {
        self.values.reverse();
    }

    pub fn reversed(&self) -> Sequence<T>
    where
        T: Clone,
    {
        Self {
            values: self.values.iter().cloned().rev().collect(),
        }
    }
}

impl<T: CubeType> Sequence<T> {
    /// Create a new empty sequence.
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Push a new value into the sequence.
    pub fn push(&mut self, value: T) {
        self.values.push(value);
    }

    /// Obtain the sequence length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn index(&self, index: usize) -> &T {
        self.values.get(index).unwrap()
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn index_mut(&mut self, index: usize) -> &mut T {
        self.values.get_mut(index).unwrap()
    }

    /// Expand function of [new](Self::new).
    pub fn __expand_new(_scope: &Scope) -> SequenceExpand<T> {
        SequenceExpand { values: Vec::new() }
    }

    /// Insert an item at the given index.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn insert(&mut self, index: usize, value: T) {
        *self.index_mut(index) = value;
    }

    /// Expand function of [push](Self::push).
    pub fn __expand_push(scope: &Scope, expand: &mut SequenceExpand<T>, value: T::ExpandType) {
        expand.__expand_push_method(scope, value)
    }

    /// Expand function of [index](Self::index).
    pub fn __expand_index<'a>(
        scope: &Scope,
        expand: &'a SequenceExpand<T>,
        index: NativeExpand<usize>,
    ) -> &'a T::ExpandType {
        expand.__expand_index_method(scope, index)
    }

    /// Expand function of [`index_mut`](Self::index_mut).
    pub fn __expand_index_mut<'a>(
        scope: &Scope,
        expand: &'a mut SequenceExpand<T>,
        index: NativeExpand<usize>,
    ) -> &'a mut T::ExpandType {
        expand.__expand_index_mut_method(scope, index)
    }
}

impl<T: CubeType> AsRefExpand for SequenceExpand<T> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl<T: CubeType> AsMutExpand for SequenceExpand<T> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<T: CubeType> Index<usize> for Sequence<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.values.index(index)
    }
}

impl<T: CubeType> IndexMut<usize> for Sequence<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.values.index_mut(index)
    }
}

impl<T: CubeType> Deref for Sequence<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T: CubeType> IndexExpand<NativeExpand<usize>> for SequenceExpand<T> {
    type Output = T::ExpandType;

    fn __expand_index_method(&self, scope: &Scope, index: NativeExpand<usize>) -> &Self::Output {
        self.__expand_index_method(scope, index)
    }
}

impl<T: CubeType> IndexMutExpand<NativeExpand<usize>> for SequenceExpand<T> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut T::ExpandType {
        self.__expand_index_mut_method(scope, index)
    }
}

/// Expand type of [Sequence].
#[derive(Debug)]
pub struct SequenceExpand<T: CubeType> {
    // We clone the expand type during the compilation phase, but for register reuse, not for
    // copying data. To achieve the intended behavior, we have to share the same underlying values.
    pub(super) values: Vec<T::ExpandType>,
}

impl<T: CubeType> Iterable for SequenceExpand<T> {
    type Item = T::ExpandType;

    fn expand(self, scope: &Scope, func: impl FnMut(&Scope, <T as CubeType>::ExpandType)) {
        self.expand_unroll(scope, func);
    }

    fn expand_unroll(
        self,
        scope: &Scope,
        mut func: impl FnMut(&Scope, <T as CubeType>::ExpandType),
    ) {
        for elem in self {
            func(scope, elem);
        }
    }

    fn const_len(&self) -> Option<usize> {
        Some(self.values.len())
    }
}

impl<T: CubeType> IntoMut for SequenceExpand<T> {
    fn into_mut(self, scope: &Scope) -> Self {
        Self {
            values: self
                .values
                .into_iter()
                .map(|v| IntoMut::into_mut(v, scope))
                .collect(),
        }
    }
}
impl<T: CubeType> CubeDebug for SequenceExpand<T> {}

impl<T: CubeType<ExpandType: Clone>> Clone for SequenceExpand<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
        }
    }
}
impl<T: CubeType> ExpandTypeClone for SequenceExpand<T> {
    fn clone_unchecked(&self) -> Self {
        Self {
            values: self.values.iter().map(|it| it.clone_unchecked()).collect(),
        }
    }
}

impl<T: CubeType> IntoIterator for Sequence<T> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<T: CubeType> IntoIterator for SequenceExpand<T> {
    type Item = T::ExpandType;

    type IntoIter = <Vec<T::ExpandType> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<T: CubeType<ExpandType: Clone>> SequenceExpand<T> {
    /// Provides an iterator without modifying the sequence
    pub fn iter_cloned(&self) -> impl Iterator<Item = T::ExpandType> {
        self.values.clone().into_iter()
    }
}

impl<T: CubeType> CubeType for Sequence<T> {
    type ExpandType = SequenceExpand<T>;
}

impl<T: CubeType> SequenceExpand<T> {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.values.len()
    }
    /// Expand method of [push](Sequence::push).
    pub fn __expand_push_method(&mut self, _scope: &Scope, value: T::ExpandType) {
        self.values.push(value);
    }

    /// Expand method of [insert](Sequence::insert).
    pub fn __expand_insert_method(&mut self, _scope: &Scope, index: usize, value: T::ExpandType) {
        if self.values.len() == index {
            self.values.push(value);
        } else {
            self.values[index] = value;
        }
    }

    /// Expand method of [index](Sequence::index).
    pub fn __expand_index_method(
        &self,
        _scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &T::ExpandType {
        let index = index.constant().expect("Index must be constant").as_usize();
        &self.values[index]
    }

    /// Expand method of [`index_mut`](Sequence::index_mut).
    pub fn __expand_index_mut_method(
        &mut self,
        _scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut T::ExpandType {
        let index = index.constant().expect("Index must be constant").as_usize();
        &mut self.values[index]
    }

    pub fn __expand_len_method(&self, _scope: &Scope) -> usize {
        self.values.len()
    }

    pub fn __expand_reverse_method(&mut self, _scope: &Scope) {
        self.values.reverse();
    }

    pub fn __expand_reversed_method(&self, _scope: &Scope) -> Self
    where
        T::ExpandType: Clone,
    {
        Self {
            values: self.values.iter().cloned().rev().collect(),
        }
    }

    pub fn __expand_clone_method(&self, _scope: &Scope) -> Self
    where
        T::ExpandType: Clone,
    {
        self.clone()
    }
}

#[macro_export]
macro_rules! seq {
    ($($value: expr),*) => {
        $crate::seq![$($value,)*]
    };
    ($($value: expr,)*) => {{
        let mut seq = Sequence::new();
        $(seq.push($value);)*
        seq
    }}
}

#[macro_export]
macro_rules! __expand_seq {
    ($scope: expr, $($value: expr),*) => {
        $crate::__expand_seq![$scope, $($value,)*]
    };
    ($scope: expr, $($value: expr,)*) => {{
        let mut seq = Sequence::__expand_new($scope);
        $(seq.__expand_push_method($scope, $value.into());)*
        seq
    }}
}
