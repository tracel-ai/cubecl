use std::num::NonZero;

use half::{bf16, f16};

use super::{
    ExpandElement, ExpandElementBaseInit, ExpandElementTyped, LaunchArgExpand, ScalarArgSettings,
    __expand_new, __expand_vectorized, init_expand_element,
};
use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::Vectorization,
};
use crate::{
    frontend::{Ceil, Cos, Erf, Exp, Floor, Log, Log1p, Powf, Recip, Sin, Sqrt, Tanh},
    ir::Item,
};
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType, Numeric},
    ir::Elem,
};
use crate::{ir::FloatKind, Runtime};

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
    + Into<Self::ExpandType>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::cmp::PartialOrd
    + std::cmp::PartialEq
{
    fn new(val: f32) -> Self;
    fn vectorized(val: f32, vectorization: u32) -> Self;
    fn vectorized_empty(vectorization: u32) -> Self;
    fn __expand_new(context: &mut CubeContext, val: f32) -> <Self as CubeType>::ExpandType {
        __expand_new(context, val)
    }
    fn __expand_vectorized(
        context: &mut CubeContext,
        val: f32,
        vectorization: u32,
    ) -> <Self as CubeType>::ExpandType {
        __expand_vectorized(context, val, vectorization, Self::as_elem())
    }

    fn __expand_vectorized_empty(
        context: &mut CubeContext,
        vectorization: u32,
    ) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_float {
    (half $primitive:ident, $kind:ident) => {
        impl_float!($primitive, $kind, |val| $primitive::from_f32(val));
    };
    ($primitive:ident, $kind:ident) => {
        impl_float!($primitive, $kind, |val| val as $primitive);
    };
    ($primitive:ident, $kind:ident, $new:expr) => {
        impl CubeType for $primitive {
            type ExpandType = ExpandElementTyped<$primitive>;
        }

        impl CubePrimitive for $primitive {
            /// Return the element type to use on GPU
            fn as_elem() -> Elem {
                Elem::Float(FloatKind::$kind)
            }
        }

        impl Numeric for $primitive {}

        impl ExpandElementBaseInit for $primitive {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl Float for $primitive {
            fn new(val: f32) -> Self {
                $new(val)
            }

            fn vectorized(val: f32, _vectorization: u32) -> Self {
                Self::new(val)
            }

            fn vectorized_empty(vectorization: u32) -> Self {
                Self::vectorized(0., vectorization)
            }

            fn __expand_vectorized_empty(
                context: &mut CubeContext,
                vectorization: u32,
            ) -> <Self as CubeType>::ExpandType {
                context
                    .create_local(Item::vectorized(
                        Self::as_elem(),
                        NonZero::new(vectorization as u8),
                    ))
                    .into()
            }
        }

        impl LaunchArgExpand for $primitive {
            fn expand(
                builder: &mut KernelBuilder,
                vectorization: Vectorization,
            ) -> ExpandElementTyped<Self> {
                assert_eq!(vectorization, None, "Attempted to vectorize a scalar");
                builder.scalar($primitive::as_elem()).into()
            }
        }
    };
}

impl_float!(half f16, F16);
impl_float!(half bf16, BF16);
impl_float!(f32, F32);
impl_float!(f64, F64);

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
