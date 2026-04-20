use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    Runtime,
    ir::{ComplexKind, ElemType, ManagedVariable, Scope, StorageType, Type},
    prelude::{CubePrimitive, CubeType, IntoRuntime, NativeAssign, NativeExpand, Scalar},
    unexpanded,
};
use cubecl_ir::{Arithmetic, ConstantValue, Operator, features::ComplexUsage};
use cubecl_runtime::client::ComputeClient;

use crate::frontend::{
    Abs,
    operation::{unary_expand, unary_expand_fixed_output},
};

pub trait ComplexCore:
    CubePrimitive
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Copy
    + Clone
    + PartialEq
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    type FloatElem: Scalar;

    fn conj(self) -> Self {
        unexpanded!()
    }

    fn real_val(self) -> Self::FloatElem {
        unexpanded!()
    }

    fn imag_val(self) -> Self::FloatElem {
        unexpanded!()
    }

    fn supported_complex_uses<R: Runtime>(
        client: &ComputeClient<R>,
    ) -> enumset::EnumSet<ComplexUsage> {
        client
            .properties()
            .complex_usage(Self::as_type_native_unchecked().storage_type())
    }
}

pub trait ComplexCompare: ComplexCore {}

pub trait ComplexMath:
    ComplexCore
    + Abs<AbsElem = Self::FloatElem>
    + crate::frontend::Exp
    + crate::frontend::Log
    + crate::frontend::Sin
    + crate::frontend::Cos
    + crate::frontend::Sqrt
    + crate::frontend::Tanh
    + crate::frontend::Powf
{
}

pub trait ComplexCoreExpand {
    fn __expand_conj_method(self, scope: &mut Scope) -> Self;
    fn __expand_real_val_method(
        self,
        scope: &mut Scope,
    ) -> NativeExpand<<Self as ComplexCoreExpand>::FloatElem>;
    fn __expand_imag_val_method(
        self,
        scope: &mut Scope,
    ) -> NativeExpand<<Self as ComplexCoreExpand>::FloatElem>;

    type FloatElem: Scalar;
}

impl<T: ComplexCore> ComplexCoreExpand for NativeExpand<T> {
    type FloatElem = T::FloatElem;

    fn __expand_conj_method(self, scope: &mut Scope) -> Self {
        unary_expand(scope, self.into(), Arithmetic::Conj).into()
    }

    fn __expand_real_val_method(self, scope: &mut Scope) -> NativeExpand<T::FloatElem> {
        let expand_element: ManagedVariable = self.into();
        let item = <T::FloatElem as CubePrimitive>::as_type(scope);
        unary_expand_fixed_output(scope, expand_element, item, Operator::Real).into()
    }

    fn __expand_imag_val_method(self, scope: &mut Scope) -> NativeExpand<T::FloatElem> {
        let expand_element: ManagedVariable = self.into();
        let item = <T::FloatElem as CubePrimitive>::as_type(scope);
        unary_expand_fixed_output(scope, expand_element, item, Operator::Imag).into()
    }
}

macro_rules! impl_complex {
    ($primitive:ty, $kind:ident, $float:ty) => {
        impl CubeType for $primitive {
            type ExpandType = NativeExpand<$primitive>;
        }

        impl CubePrimitive for $primitive {
            type Scalar = Self;
            type Size = crate::prelude::Const<1>;
            type WithScalar<S: Scalar> = S;

            fn as_type_native() -> Option<Type> {
                Some(StorageType::Scalar(ElemType::Complex(ComplexKind::$kind)).into())
            }

            fn from_const_value(value: ConstantValue) -> Self {
                let ConstantValue::Complex(re, im) = value else {
                    unreachable!("expected Complex constant")
                };
                <$primitive>::new(re as $float, im as $float)
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
                self.into()
            }
        }

        impl NativeAssign for $primitive {}

        impl crate::prelude::IntoMut for $primitive {
            fn into_mut(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl Scalar for $primitive {}

        impl Abs for $primitive {
            type AbsElem = $float;
        }

        impl ComplexCore for $primitive {
            type FloatElem = $float;
        }

        impl ComplexCompare for $primitive {}

        impl ComplexMath for $primitive {}
    };
}

impl_complex!(num_complex::Complex<f32>, C32, f32);
impl_complex!(num_complex::Complex<f64>, C64, f64);
