use std::num::NonZero;

use crate::{
    ir::{Elem, FloatKind, IntKind},
    prelude::{UInt, F32, F64, I32, I64},
};

use super::{Expr, Expression};

pub trait TypeEq<T> {}
impl<T> TypeEq<T> for T {}

pub trait SquareType {
    fn ir_type() -> Elem;
    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

pub trait Primitive: SquareType {
    fn value(&self) -> PrimitiveValue;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimitiveValue {
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Unit,
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

pub trait Integer: Clone {}
pub trait KernelArg {}

impl<T: SquareType> KernelArg for T {}

pub trait Expand: Sized {
    type Expanded<Inner: Expr<Output = Self>>;

    fn expand<Inner: Expr<Output = Self>>(base: Inner) -> Self::Expanded<Inner>;
}

pub trait StaticExpand: Sized {
    type Expanded;
}

/// Auto impl `StaticExpand for all `Expand` types, with `Self` as the inner expression
impl<T: Expand + Expr<Output = T>> StaticExpand for T {
    type Expanded = <T as Expand>::Expanded<T>;
}

pub trait ExpandExpr<Inner: Expand>: Expr<Output = Inner> + Sized {
    fn expand(self) -> Inner::Expanded<Self> {
        Inner::expand(self)
    }
}

impl<Expression: Expr> ExpandExpr<Expression::Output> for Expression where Expression::Output: Expand
{}

pub trait MethodExpand: Sized {}

impl SquareType for () {
    fn ir_type() -> Elem {
        Elem::Unit
    }
}

impl Primitive for () {
    fn value(&self) -> PrimitiveValue {
        PrimitiveValue::Unit
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

macro_rules! vectorized_primitive {
    ($primitive:ident, $var_type:expr) => {
        impl SquareType for $primitive {
            fn ir_type() -> Elem {
                $var_type
            }

            fn vectorization(&self) -> Option<NonZero<u8>> {
                NonZero::new(self.vectorization)
            }
        }
    };
}

macro_rules! int_primitive {
    ($primitive:ident, $var_type:expr) => {
        primitive!($primitive, $var_type);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> PrimitiveValue {
                PrimitiveValue::Int(*self as i64)
            }
        }
    };
}

macro_rules! uint_primitive {
    ($primitive:ident, $var_type:expr) => {
        primitive!($primitive, $var_type);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> PrimitiveValue {
                PrimitiveValue::UInt(*self as u64)
            }
        }
    };
}

macro_rules! float_primitive {
    ($primitive:ident, $var_type:expr) => {
        primitive!($primitive, $var_type);

        impl Primitive for $primitive {
            fn value(&self) -> PrimitiveValue {
                PrimitiveValue::Float(*self as f64)
            }
        }
    };
}

macro_rules! vectorized_int_primitive {
    ($primitive:ident, $var_type:expr) => {
        vectorized_primitive!($primitive, $var_type);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> PrimitiveValue {
                PrimitiveValue::Int(self.val as i64)
            }
        }
    };
}

macro_rules! vectorized_uint_primitive {
    ($primitive:ident, $var_type:expr) => {
        vectorized_primitive!($primitive, $var_type);

        impl Integer for $primitive {}
        impl Primitive for $primitive {
            fn value(&self) -> PrimitiveValue {
                PrimitiveValue::UInt(self.val as u64)
            }
        }
    };
}

macro_rules! vectorized_float_primitive {
    ($primitive:ident, $var_type:expr) => {
        vectorized_primitive!($primitive, $var_type);

        impl Primitive for $primitive {
            fn value(&self) -> PrimitiveValue {
                PrimitiveValue::Float(self.val as f64)
            }
        }
    };
}

int_primitive!(i32, Elem::Int(IntKind::I32));
int_primitive!(i64, Elem::Int(IntKind::I64));
uint_primitive!(u32, Elem::UInt);
float_primitive!(f32, Elem::Float(FloatKind::F32));
float_primitive!(f64, Elem::Float(FloatKind::F64));

vectorized_uint_primitive!(UInt, Elem::UInt);
vectorized_int_primitive!(I32, Elem::Int(IntKind::I32));
vectorized_int_primitive!(I64, Elem::Int(IntKind::I64));
vectorized_float_primitive!(F32, Elem::Float(FloatKind::F32));
vectorized_float_primitive!(F64, Elem::Float(FloatKind::F64));

primitive!(bool, Elem::Bool);

impl Primitive for bool {
    fn value(&self) -> PrimitiveValue {
        PrimitiveValue::Bool(*self)
    }
}
