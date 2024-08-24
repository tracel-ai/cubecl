use std::num::NonZero;

use crate::{
    ir::{Elem, FloatKind, IntKind},
    prelude::{UInt, F32, F64, I32, I64},
};

use super::Expr;

pub trait TypeEq<T> {}
impl<T> TypeEq<T> for T {}

pub trait SquareType {
    fn ir_type() -> Elem;
    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

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
        Elem::Pointer
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
    };
}

macro_rules! vectorized_int_primitive {
    ($primitive:ident, $var_type:expr) => {
        vectorized_primitive!($primitive, $var_type);
    };
}

int_primitive!(i32, Elem::Int(IntKind::I32));
int_primitive!(i64, Elem::Int(IntKind::I64));
int_primitive!(u32, Elem::UInt);
primitive!(f32, Elem::Float(FloatKind::F32));
primitive!(f64, Elem::Float(FloatKind::F64));

vectorized_int_primitive!(UInt, Elem::UInt);
vectorized_int_primitive!(I32, Elem::Int(IntKind::I32));
vectorized_int_primitive!(I64, Elem::Int(IntKind::I64));
vectorized_primitive!(F32, Elem::Float(FloatKind::F32));
vectorized_primitive!(F64, Elem::Float(FloatKind::F64));

primitive!(bool, Elem::Bool);

// impl NumCast for UInt {
//     fn from<T: ToPrimitive>(n: T) -> Option<Self> {
//         n.to_u32().map(Into::into)
//     }
// }

// impl ToPrimitive for UInt {
//     fn to_i64(&self) -> Option<i64> {
//         Some(self.val as i64)
//     }

//     fn to_u64(&self) -> Option<u64> {
//         Some(self.val as u64)
//     }
// }

// impl Num for UInt {
//     type FromStrRadixErr = <u32 as Num>::FromStrRadixErr;

//     fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
//         u32::from_str_radix(str, radix).map(Into::into)
//     }
// }

// impl One for UInt {
//     fn one() -> Self {
//         1.into()
//     }
// }

// impl Zero for UInt {
//     fn zero() -> Self {
//         0.into()
//     }

//     fn is_zero(&self) -> bool {
//         self.val == 0
//     }
// }
