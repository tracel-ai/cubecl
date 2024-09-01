use crate::{
    ir::Elem,
    new_ir::{Expr, Expression, OnceExpr, SquareType, StaticExpand, StaticExpanded},
    unexpanded,
};
use std::{
    cell::RefCell,
    mem,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use super::Integer;

/// A sequence of [cube types](CubeType) that is inlined during compilation.
///
/// In other words, it allows you to group a dynamic amount of variables at compile time.
///
/// All methods [push](Sequence::push), [index](Sequence::index) and
/// [into_iter](Sequence::into_iter) are executed _during_ compilation and don't add any overhead
/// on the generated kernel.
pub struct Sequence<T: SquareType> {
    values: RefCell<Vec<T>>,
}

/// Expand type of [Sequence].
pub struct SequenceExpand<T: SquareType> {
    // We clone the expand type during the compilation phase, but for register reuse, not for
    // copying data. To achieve the intended behavior, we have to share the same underlying values.
    values: Rc<RefCell<Vec<OnceExpr<T>>>>,
}

impl<T: SquareType> StaticExpanded for SequenceExpand<T> {
    type Unexpanded = Sequence<T>;
}

impl<T: SquareType> StaticExpand for Sequence<T> {
    type Expanded = SequenceExpand<T>;
}

impl<T: SquareType> Expr for Sequence<T> {
    type Output = Self;
    fn expression_untyped(&self) -> Expression {
        panic!("Can't expand struct directly");
    }
    fn vectorization(&self) -> Option<::core::num::NonZero<u8>> {
        None
    }
}
impl<T: SquareType> Expr for &Sequence<T> {
    type Output = Self;
    fn expression_untyped(&self) -> Expression {
        panic!("Can't expand struct directly");
    }
    fn vectorization(&self) -> Option<::core::num::NonZero<u8>> {
        None
    }
}
impl<T: SquareType> Expr for &mut Sequence<T> {
    type Output = Self;
    fn expression_untyped(&self) -> Expression {
        panic!("Can't expand struct directly");
    }
    fn vectorization(&self) -> Option<::core::num::NonZero<u8>> {
        None
    }
}
impl<T: SquareType> SquareType for Sequence<T> {
    fn ir_type() -> Elem {
        T::ir_type()
    }
}

impl<T: SquareType> Default for Sequence<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: SquareType> Send for Sequence<T> {}
unsafe impl<T: SquareType> Sync for Sequence<T> {}

impl<T: SquareType> Sequence<T> {
    /// Create a new empty sequence.
    pub fn new() -> Self {
        Self {
            values: Vec::new().into(),
        }
    }

    /// Push a new value into the sequence.
    pub fn push(&self, value: T) {
        self.values.borrow_mut().push(value);
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn index<I: Integer>(&self, index: I) -> &T {
        unexpanded!();
    }
}

impl<T: SquareType> SequenceExpand<T> {
    /// Expand function of [new](Self::new).
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> SequenceExpand<T> {
        SequenceExpand {
            values: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl<T: SquareType> Default for SequenceExpand<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SquareType> SequenceExpand<T> {
    pub fn expand(&self) -> &Self {
        self
    }
}

impl<T: SquareType> Clone for SequenceExpand<T> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
        }
    }
}

impl<T: SquareType> IntoIterator for Sequence<T> {
    type Item = T;

    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let values = mem::take(self.values.borrow_mut().deref_mut());
        values.into_iter()
    }
}

impl<T: SquareType> IntoIterator for SequenceExpand<T> {
    type Item = OnceExpr<T>;

    type IntoIter = <Vec<OnceExpr<T>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.take().into_iter()
    }
}

impl<T: SquareType> SequenceExpand<T> {
    /// Expand method of [push](Sequence::push).
    pub fn push(&self, value: impl Expr<Output = T> + 'static) {
        self.values.deref().borrow_mut().push(OnceExpr::new(value));
    }

    /// Expand method of [index](Sequence::index).
    pub fn index<I: Integer>(&self, index: impl Expr<Output = I>) -> impl Expr<Output = T> {
        let index = index
            .expression_untyped()
            .as_lit()
            .expect("Only constant are supported")
            .as_usize();
        self.values.borrow()[index].clone()
    }
}

impl<T: SquareType> SquareType for SequenceExpand<T> {
    fn ir_type() -> Elem {
        T::ir_type()
    }
}
