use cubecl_macros_2::{expand_impl, Expand};

use crate::{
    ir::Elem,
    new_ir::{DynamicExpr, Expr, Integer, RcExpr, SquareType, Variable},
    unexpanded,
};
use std::{cell::RefCell, rc::Rc};

/// A sequence of [cube types](CubeType) that is inlined during compilation.
///
/// In other words, it allows you to group a dynamic amount of variables at compile time.
///
/// All methods [push](Sequence::push), [index](Sequence::index) and
/// [into_iter](Sequence::into_iter) are executed _during_ compilation and don't add any overhead
/// on the generated kernel.
#[derive(Expand)]
#[expand(ir_type = T::ir_type())]
pub struct Sequence<T: SquareType> {
    #[expand(skip)]
    values: Vec<T>,
}

impl<T: SquareType> Default for Sequence<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: SquareType> Send for Sequence<T> {}
unsafe impl<T: SquareType> Sync for Sequence<T> {}

#[expand_impl]
impl<T: SquareType> Sequence<T> {
    /// Create a new empty sequence.
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Push a new value into the sequence.
    pub fn push(&mut self, value: T) {
        self.values.push(value);
    }

    /// Get the variable at the given position in the sequence.
    #[allow(unused_variables, clippy::should_implement_trait)]
    pub fn index<I: Integer>(&self, index: I) -> &T {
        unexpanded!();
    }

    /// Expand function of [new](Self::new).
    #[expanded]
    pub fn new() -> SequenceExpanded<T> {
        SequenceExpanded {
            values: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

/// Expand type of [Sequence].
pub struct SequenceExpanded<T: SquareType> {
    // We clone the expand type during the compilation phase, but for register reuse, not for
    // copying data. To achieve the intended behavior, we have to share the same underlying values.
    values: Rc<RefCell<Vec<RcExpr<T>>>>,
}

impl<T: SquareType> Expr for SequenceExpanded<T> {
    type Output = Self;

    fn expression_untyped(&self) -> crate::new_ir::Expression {
        todo!()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        todo!()
    }
}

impl<T: SquareType> SequenceExpanded<T> {
    pub fn expand(&self) -> &Self {
        self
    }
}

impl<T: SquareType> Clone for SequenceExpanded<T> {
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
        self.values.into_iter()
    }
}

impl<T: SquareType> IntoIterator for SequenceExpanded<T> {
    type Item = RcExpr<T>;

    type IntoIter = <Vec<RcExpr<T>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.take().into_iter()
    }
}

impl<T: SquareType> SequenceExpanded<T> {
    /// Expand method of [push](Sequence::push).
    pub fn push(&mut self, value: impl Expr<Output = T> + 'static) {
        self.values.borrow_mut().push(RcExpr::new(value));
    }

    /// Expand method of [index](Sequence::index).
    pub fn index<I: Integer>(&self, index: impl Expr<Output = T>) -> impl Expr<Output = T> {
        let index = index
            .expression_untyped()
            .as_lit()
            .expect("Only constant are supported")
            .as_usize();
        self.values.borrow()[index].clone()
    }
}

impl<T: SquareType> SquareType for SequenceExpanded<T> {
    fn ir_type() -> Elem {
        T::ir_type()
    }
}
