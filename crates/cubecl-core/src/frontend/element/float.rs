use half::{bf16, f16};

use crate::frontend::{Ceil, Cos, Erf, Exp, Floor, Log, Log1p, Powf, Recip, Sin, Sqrt, Tanh};
use crate::frontend::{
    ComptimeType, CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementBaseInit,
    ExpandElementTyped, Numeric,
};
use crate::ir::{ConstantScalarValue, Elem, FloatKind, Variable, Vectorization};

use super::{
    init_expand_element, LaunchArgExpand, ScalarArgSettings, UInt, Vectorized, __expand_new,
    __expand_vectorized,
};
use crate::compute::{KernelBuilder, KernelLauncher};
use crate::Runtime;

/// Floating point numbers. Used as input in float kernels
pub trait Float:
    Numeric
    + Exp
    + Log
    + Log1p
    + Cos
    + Sin
    + Tanh
    + Powf
    + Sqrt
    + Floor
    + Ceil
    + Erf
    + Recip
    + From<f32>
    + core::ops::Add<f32, Output = Self>
    + core::ops::Sub<f32, Output = Self>
    + core::ops::Mul<f32, Output = Self>
    + core::ops::Div<f32, Output = Self>
    + std::ops::AddAssign<f32>
    + std::ops::SubAssign<f32>
    + std::ops::MulAssign<f32>
    + std::ops::DivAssign<f32>
    + std::cmp::PartialOrd<f32>
    + std::cmp::PartialEq<f32>
{
    fn new(val: f32) -> Self;
    fn vectorized(val: f32, vectorization: UInt) -> Self;
    fn __expand_new(
        context: &mut CubeContext,
        val: Self::ExpandType,
    ) -> <Self as CubeType>::ExpandType {
        __expand_new(context, val, Self::as_elem())
    }
    fn __expand_vectorized(
        context: &mut CubeContext,
        val: Self::ExpandType,
        vectorization: UInt,
    ) -> <Self as CubeType>::ExpandType {
        __expand_vectorized(context, val, vectorization, Self::as_elem())
    }
}

macro_rules! impl_float {
    ($type:ident, $primitive:ty) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f32,
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<$type>;
        }

        impl CubePrimitive for $type {
            /// Return the element type to use on GPU
            fn as_elem() -> Elem {
                Elem::Float(FloatKind::$type)
            }
        }

        impl ComptimeType for $type {
            fn into_expand(self) -> Self::ExpandType {
                let elem = Self::as_elem();
                let value = self.val as f64;
                let value = match elem {
                    Elem::Float(kind) => ConstantScalarValue::Float(value, kind),
                    _ => panic!("Wrong elem type"),
                };

                ExpandElementTyped::new(ExpandElement::Plain(Variable::ConstantScalar(value)))
            }
        }

        impl From<$type> for ExpandElement {
            fn from(value: $type) -> Self {
                let constant = $type::as_elem().from_constant(value.val.into());
                ExpandElement::Plain(constant)
            }
        }

        impl Numeric for $type {
            type Primitive = $primitive;
        }

        impl From<u32> for $type {
            fn from(val: u32) -> Self {
                $type::from_int(val)
            }
        }

        impl ExpandElementBaseInit for $type {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl Float for $type {
            fn new(val: f32) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn vectorized(val: f32, vectorization: UInt) -> Self {
                if vectorization.val == 1 {
                    Self::new(val)
                } else {
                    Self {
                        val,
                        vectorization: vectorization.val as u8,
                    }
                }
            }
        }

        impl LaunchArgExpand for $type {
            fn expand(
                builder: &mut KernelBuilder,
                vectorization: Vectorization,
            ) -> ExpandElementTyped<Self> {
                assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
                builder.scalar($type::as_elem()).into()
            }
        }

        impl Vectorized for $type {
            fn vectorization_factor(&self) -> UInt {
                UInt {
                    val: self.vectorization as u32,
                    vectorization: 1,
                }
            }

            fn vectorize(mut self, factor: UInt) -> Self {
                self.vectorization = factor.vectorization;
                self
            }
        }
    };
}

impl_float!(F16, f16);
impl_float!(BF16, bf16);
impl_float!(F32, f32);
impl_float!(F64, f64);

impl From<f32> for F32 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<f32> for BF16 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<f32> for F16 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<f32> for F64 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl ScalarArgSettings for f16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f16(*self);
    }
}

impl ScalarArgSettings for bf16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_bf16(*self);
    }
}

impl ScalarArgSettings for f32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(*self);
    }
}

impl ScalarArgSettings for f64 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f64(*self);
    }
}
