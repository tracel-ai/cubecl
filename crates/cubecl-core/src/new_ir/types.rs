use super::{Expr, Expression, UnaryOp};
use crate::ir::{ConstantScalarValue, Elem, FloatKind, IntKind};
use cubecl_common::operator::Operator;
use half::{bf16, f16};
use num_traits::{NumAssign, NumCast, ToPrimitive};
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

pub trait Primitive: SquareType + 'static {
    fn value(&self) -> ConstantScalarValue;
}

impl<T: Primitive> Expr for T {
    type Output = T;

    fn expression_untyped(&self) -> super::Expression {
        Expression::Literal {
            value: self.value(),
            vectorization: self.vectorization(),
            ty: <T as SquareType>::ir_type(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.vectorization()
    }
}

pub trait KernelArg {}

impl<T: SquareType> KernelArg for T {}

/// Type that has runtime fields or methods
pub trait Expand: Sized {
    type Expanded<Inner: Expr<Output = Self>>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner>;
}

/// Comptime type that has fields or methods that create runtime values (i.e. `Option<SquareType>`)
pub trait PartialExpand: Sized {
    type Expanded;

    fn partial_expand(self) -> Self::Expanded;
}

/// Type that has associated functions to expand into runtime functions
pub trait StaticExpand: Sized {
    type Expanded;
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

pub trait ExpandExpr<Inner: Expand>: Expr<Output = Inner> + Sized {
    fn expand(self) -> Inner::Expanded<Self> {
        Inner::expand(self)
    }
}

impl<Expression: Expr> ExpandExpr<Expression::Output> for Expression where Expression::Output: Expand
{}

pub trait MethodExpand: Sized {}

pub trait Numeric:
    Primitive
    + NumCast
    + NumAssign
    + PartialOrd
    + PartialEq
    + Expand<Expanded<Self> = NumericExpand<Self>>
{
    fn new<N: ToPrimitive>(n: N) -> Self {
        <Self as NumCast>::from(n).unwrap()
    }
}
pub trait Float: Numeric + num_traits::Float {}
pub trait Integer: Numeric {}

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

pub struct NumericExpand<Inner: Expr>(Inner)
where
    Inner::Output: Numeric;

impl<Inner: NumCast + Expr> NumericExpand<Inner>
where
    Inner::Output: Numeric,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new<N: ToPrimitive>(n: N) -> Inner {
        <Inner as NumCast>::from(n).unwrap()
    }
}

#[derive(new)]
pub struct CosExpr<In: Expr>(pub UnaryOp<In, In::Output>)
where
    In::Output: Float;

impl<In: Expr> Expr for CosExpr<In>
where
    In::Output: Float,
{
    type Output = In::Output;

    fn expression_untyped(&self) -> Expression {
        Expression::Unary {
            input: Box::new(self.0.input.expression_untyped()),
            operator: Operator::Cos,
            vectorization: self.vectorization(),
            ty: In::Output::ir_type(),
        }
    }

    fn vectorization(&self) -> Option<NonZero<u8>> {
        self.0.input.vectorization()
    }
}

impl<Inner: Expr> NumericExpand<Inner>
where
    Inner::Output: Float,
{
    pub fn cos(num: impl Expr<Output = Inner::Output>) -> impl Expr<Output = Inner::Output> {
        CosExpr(UnaryOp::new(num))
    }
}

macro_rules! primitive {
    ($primitive:ident, $var_type:expr) => {
        impl SquareType for $primitive {
            fn ir_type() -> Elem {
                $var_type
            }
        }
    };
}

macro_rules! numeric_primitive {
    ($primitive:ident, $var_type:expr) => {
        primitive!($primitive, $var_type);

        impl Numeric for $primitive {}
        impl Expand for $primitive {
            type Expanded<Inner: Expr<Output = Self>> = NumericExpand<Inner>;

            fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
                NumericExpand(inner)
            }
        }
    };
}

macro_rules! int_primitive {
    ($primitive:ident, $var_type:expr, $kind:expr) => {
        numeric_primitive!($primitive, $var_type($kind));

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> ConstantScalarValue {
                ConstantScalarValue::Int(*self as i64, $kind)
            }
        }
    };
}

macro_rules! uint_primitive {
    ($primitive:ident, $var_type:expr) => {
        numeric_primitive!($primitive, $var_type);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> ConstantScalarValue {
                ConstantScalarValue::UInt(*self as u64)
            }
        }
    };
}

macro_rules! float_primitive {
    ($primitive:ident, $var_type:expr, $kind:expr) => {
        numeric_primitive!($primitive, $var_type($kind));

        impl Float for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> ConstantScalarValue {
                ConstantScalarValue::Float(self.to_f64().unwrap(), $kind)
            }
        }
    };
}

int_primitive!(i32, Elem::Int, IntKind::I32);
int_primitive!(i64, Elem::Int, IntKind::I64);
uint_primitive!(u32, Elem::UInt);
float_primitive!(f16, Elem::Float, FloatKind::F16);
float_primitive!(bf16, Elem::Float, FloatKind::BF16);
float_primitive!(f32, Elem::Float, FloatKind::F32);
float_primitive!(f64, Elem::Float, FloatKind::F64);
primitive!(bool, Elem::Bool);

impl Primitive for bool {
    fn value(&self) -> ConstantScalarValue {
        ConstantScalarValue::Bool(*self)
    }
}
