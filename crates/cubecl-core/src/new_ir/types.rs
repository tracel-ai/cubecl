use super::Expr;
use crate::{
    ir::{ConstantScalarValue, Elem},
    prelude::Primitive,
};
use std::num::NonZero;

pub trait TypeEq<T> {}
impl<T> TypeEq<T> for T {}

pub trait SquareType {
    fn ir_type() -> Elem;
    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

impl<T: SquareType> SquareType for &T {
    fn ir_type() -> Elem {
        T::ir_type()
    }
}

impl<T: SquareType> SquareType for &mut T {
    fn ir_type() -> Elem {
        T::ir_type()
    }
}

pub trait Container {
    type Item: SquareType;
}

/// Type that has runtime fields or methods
pub trait Expand: Sized {
    type Expanded<Inner: Expr<Output = Self>>: Expanded<Unexpanded = Self>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner>;
}

pub trait Expanded: Sized {
    type Unexpanded: Expand;
    fn inner(self) -> impl Expr<Output = Self::Unexpanded>;
}

/// Comptime type that has fields or methods that create runtime values (i.e. `Option<SquareType>`)
pub trait PartialExpand: Sized {
    type Expanded: StaticExpanded<Unexpanded = Self>;

    fn partial_expand(self) -> Self::Expanded;
}

/// Type that has associated functions to expand into runtime functions
pub trait StaticExpand: Sized {
    type Expanded: StaticExpanded<Unexpanded = Self>;
}

/// Type that has associated functions to expand into runtime functions
pub trait StaticExpanded: Sized {
    type Unexpanded;
}

/// Auto impl `StaticExpand for all `Expand` types, with `Self` as the inner expression
impl<T: PartialExpand + Expr<Output = T>> StaticExpand for T {
    type Expanded = <T as PartialExpand>::Expanded;
}

/// All fully expanded types can also be partially expanded if receiver is const
impl<T: Expand + Expr<Output = T>> PartialExpand for T {
    type Expanded = <T as Expand>::Expanded<Self>;

    fn partial_expand(self) -> Self::Expanded {
        <T as Expand>::expand(self)
    }
}

impl<T: Expanded> StaticExpanded for T {
    type Unexpanded = T::Unexpanded;
}

pub trait ExpandExpr<Inner: Expand>: Expr<Output = Inner> + Sized {
    fn expand(self) -> Inner::Expanded<Self> {
        Inner::expand(self)
    }
}

impl<Expression: Expr> ExpandExpr<Expression::Output> for Expression where Expression::Output: Expand
{}

impl SquareType for () {
    fn ir_type() -> Elem {
        Elem::Unit
    }
}

impl Primitive for () {
    fn value(&self) -> ConstantScalarValue {
        ConstantScalarValue::UInt(0)
    }
}
