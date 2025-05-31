use cubecl_ir::{ExpandElement, Scope};
use serde::{Deserialize, Serialize};

use crate::{
    frontend::{CubeType, ExpandElementTyped, IntoMut, branch::Iterable, indexation::Index},
    prelude::CubeDebug,
};
use std::{cell::RefCell, rc::Rc};

/// A sequence of [cube types](CubeType) that is inlined during compilation.
///
/// In other words, it allows you to group a dynamic amount of variables at compile time.
///
/// All methods [push](Sequence::push), [index](Sequence::index) and
/// [into_iter](Sequence::into_iter) are executed _during_ compilation and don't add any overhead
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

impl<T: CubeType> IntoMut for Sequence<T> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}
impl<T: CubeType> CubeDebug for Sequence<T> {}

impl<T: CubeType + Clone> Sequence<T> {
    pub fn rev(&self) -> Self {
        Self {
            values: self.values.iter().rev().cloned().collect(),
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
    pub fn len(&self) -> u32 {
        self.values.len() as u32
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn index<I: Index>(&self, index: I) -> &T {
        let index: ExpandElementTyped<u32> = ExpandElement::Plain(index.value()).into();
        let index = index
            .constant()
            .expect("Only constant are supported")
            .as_usize();

        self.values.get(index).unwrap()
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn index_mut<I: Index>(&mut self, index: I) -> &mut T {
        let index: ExpandElementTyped<u32> = ExpandElement::Plain(index.value()).into();
        let index = index
            .constant()
            .expect("Only constant are supported")
            .as_usize();

        self.values.get_mut(index).unwrap()
    }

    /// Expand function of [new](Self::new).
    pub fn __expand_new(_scope: &mut Scope) -> SequenceExpand<T> {
        SequenceExpand {
            values: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Insert an item at the given index.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn insert<I: Index>(&mut self, index: I, value: T) {
        *self.index_mut(index) = value;
    }

    /// Expand function of [push](Self::push).
    pub fn __expand_push(scope: &mut Scope, expand: &mut SequenceExpand<T>, value: T::ExpandType) {
        expand.__expand_push_method(scope, value)
    }

    /// Expand function of [index](Self::index).
    pub fn __expand_index(
        scope: &mut Scope,
        expand: SequenceExpand<T>,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        expand.__expand_index_method(scope, index)
    }

    /// Expand function of [index_mut](Self::index_mut).
    pub fn __expand_index_mut(
        scope: &mut Scope,
        expand: SequenceExpand<T>,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        expand.__expand_index_mut_method(scope, index)
    }
}

/// Expand type of [Sequence].
pub struct SequenceExpand<T: CubeType> {
    // We clone the expand type during the compilation phase, but for register reuse, not for
    // copying data. To achieve the intended behavior, we have to share the same underlying values.
    pub(super) values: Rc<RefCell<Vec<T::ExpandType>>>,
}

impl<T: CubeType> Iterable<T> for SequenceExpand<T> {
    fn expand(self, scope: &mut Scope, func: impl FnMut(&mut Scope, <T as CubeType>::ExpandType)) {
        self.expand_unroll(scope, func);
    }

    fn expand_unroll(
        self,
        scope: &mut Scope,
        mut func: impl FnMut(&mut Scope, <T as CubeType>::ExpandType),
    ) {
        for elem in self {
            func(scope, elem);
        }
    }
}

impl<T: CubeType> IntoMut for SequenceExpand<T> {
    fn into_mut(self, scope: &mut Scope) -> Self {
        let mut values = self.values.borrow_mut();
        values.iter_mut().for_each(|v| {
            *v = IntoMut::into_mut(v.clone(), scope);
        });
        core::mem::drop(values);

        self
    }
}
impl<T: CubeType> CubeDebug for SequenceExpand<T> {}

impl<T: CubeType> Clone for SequenceExpand<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
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
        self.values.take().into_iter()
    }
}

impl<T: CubeType> CubeType for Sequence<T> {
    type ExpandType = SequenceExpand<T>;
}

impl<T: CubeType> SequenceExpand<T> {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        self.values.borrow().len() as u32
    }
    /// Expand method of [push](Sequence::push).
    pub fn __expand_push_method(&mut self, _scope: &mut Scope, value: T::ExpandType) {
        self.values.borrow_mut().push(value);
    }

    /// Expand method of [insert](Sequence::insert).
    pub fn __expand_insert_method(
        &self,
        _scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: T::ExpandType,
    ) {
        let index = index
            .constant()
            .expect("Only constant are supported")
            .as_usize();

        let mut values = self.values.borrow_mut();

        if values.len() == index {
            values.push(value);
        } else {
            values[index] = value;
        }
    }

    /// Expand method of [index](Sequence::index).
    pub fn __expand_index_method(
        &self,
        _scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        let index = index
            .constant()
            .expect("Only constant are supported")
            .as_usize();

        self.values.borrow()[index].clone()
    }

    /// Expand method of [index_mut](Sequence::index_mut).
    pub fn __expand_index_mut_method(
        &self,
        _scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> T::ExpandType {
        let index = index
            .constant()
            .expect("Only constant are supported")
            .as_usize();

        self.values.borrow()[index].clone()
    }

    pub fn __expand_len_method(&self, _scope: &mut Scope) -> u32 {
        let values = self.values.borrow();
        values.len() as u32
    }

    pub fn __expand_rev_method(self, _scope: &mut Scope) -> Self {
        self.values.borrow_mut().reverse();
        self
    }
}
