use std::{cell::RefCell, rc::Rc};

use crate::unexpanded;

use super::{indexation::Index, CubeContext, CubeType, Init, UInt};

/// A sequence of [cube types](CubeType) that is inlined during compilation.
///
/// In other words, it allows you to group a dynamic amount of variables at compile time.
///
/// All methods [push](Sequence::push), [index](Sequence::index) and
/// [into_iter](Sequence::into_iter) are executed _during_ compilation and don't add any overhead
/// on the generated kernel.
pub struct Sequence<T: CubeType> {
    values: Vec<T>,
}

impl<T: CubeType> Sequence<T> {
    /// Create a new empty sequence.
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn __expand_new(_context: &mut CubeContext) -> SequenceExpand<T> {
        SequenceExpand {
            values: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Push a new value into the sequence.
    pub fn push(&mut self, value: T) {
        self.values.push(value);
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables)]
    pub fn index<I: super::indexation::Index>(&self, index: I) -> &T {
        unexpanded!();
    }

    /// Expand function of [push](Self::push).
    pub fn __expand_push(
        context: &mut CubeContext,
        expand: &mut SequenceExpand<T>,
        value: T::ExpandType,
    ) {
        expand.__expand_push_method(context, value)
    }

    /// Expand function of [index](Self::index).
    pub fn __expand_index(
        context: &mut CubeContext,
        expand: SequenceExpand<T>,
        index: <UInt as CubeType>::ExpandType,
    ) -> T::ExpandType {
        expand.__expand_index_method(context, index)
    }
}

/// Expand type of [Sequence].
pub struct SequenceExpand<T: CubeType> {
    values: Rc<RefCell<Vec<T::ExpandType>>>,
}

impl<T: CubeType> Init for SequenceExpand<T> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        self
    }
}

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
    /// Expand method of [push](Sequence::push).
    pub fn __expand_push_method(&mut self, _context: &mut CubeContext, value: T::ExpandType) {
        self.values.borrow_mut().push(value);
    }

    /// Expand method of [index](Sequence::index).
    pub fn __expand_index_method<I: Index>(
        &self,
        _context: &mut CubeContext,
        index: I,
    ) -> T::ExpandType {
        let value = index.value();
        let index = match value {
            crate::ir::Variable::ConstantScalar(value) => match value {
                crate::ir::ConstantScalarValue::Int(val, _) => val as usize,
                crate::ir::ConstantScalarValue::UInt(val) => val as usize,
                _ => panic!("Only integer types are supporterd"),
            },
            _ => panic!("Only constant are supported"),
        };
        self.values.borrow()[index].clone()
    }
}
