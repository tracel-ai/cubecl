use std::num::NonZero;

use crate::{
    ir::{Elem, FloatKind, IntKind},
    prelude::{UInt, F32, F64, I32, I64},
};

use super::Expr;

pub trait SquareType {
    fn ir_type() -> Elem;
    fn vectorization(&self) -> Option<NonZero<u8>> {
        None
    }
}

pub trait KernelArg {}

impl<T: SquareType> KernelArg for T {}

pub trait KernelStruct: SquareType + Sized {
    type Expanded<Base: Expr<Output = Self> + Clone>;

    fn expand<Base: Expr<Output = Self> + Clone>(base: Base) -> Self::Expanded<Base>;
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

primitive!(i32, Elem::Int(IntKind::I32));
primitive!(i64, Elem::Int(IntKind::I64));
primitive!(u32, Elem::UInt);
primitive!(f32, Elem::Float(FloatKind::F32));
primitive!(f64, Elem::Float(FloatKind::F64));

vectorized_primitive!(UInt, Elem::UInt);
vectorized_primitive!(I32, Elem::Int(IntKind::I32));
vectorized_primitive!(I64, Elem::Int(IntKind::I64));
vectorized_primitive!(F32, Elem::Float(FloatKind::F32));
vectorized_primitive!(F64, Elem::Float(FloatKind::F64));

primitive!(bool, Elem::Bool);
