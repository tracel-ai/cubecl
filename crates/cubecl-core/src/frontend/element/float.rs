use cubecl_ir::Scope;
use half::{bf16, f16};

use crate::{
    ir::{Elem, ExpandElement, FloatKind},
    prelude::*,
};

use super::Numeric;

mod fp4;
mod fp6;
mod fp8;
mod relaxed;
mod tensor_float;
mod typemap;

pub use typemap::*;

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
    + Round
    + Floor
    + Ceil
    + Erf
    + Recip
    + Magnitude
    + Normalize
    + Dot
    + Into<Self::ExpandType>
    + core::ops::Neg<Output = Self>
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
    const DIGITS: u32;
    const EPSILON: Self;
    const INFINITY: Self;
    const MANTISSA_DIGITS: u32;
    const MAX_10_EXP: i32;
    const MAX_EXP: i32;
    const MIN_10_EXP: i32;
    const MIN_EXP: i32;
    const MIN_POSITIVE: Self;
    const NAN: Self;
    const NEG_INFINITY: Self;
    const RADIX: u32;

    fn new(val: f32) -> Self;
    fn __expand_new(scope: &mut Scope, val: f32) -> <Self as CubeType>::ExpandType {
        __expand_new(scope, val)
    }
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
            fn as_elem_native() -> Option<Elem> {
                Some(Elem::Float(FloatKind::$kind))
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
                let elem: ExpandElementTyped<Self> = self.into();
                into_runtime_expand_element(scope, elem).into()
            }
        }

        impl Numeric for $primitive {
            fn min_value() -> Self {
                <Self as num_traits::Float>::min_value()
            }
            fn max_value() -> Self {
                <Self as num_traits::Float>::max_value()
            }
        }

        impl ExpandElementIntoMut for $primitive {
            fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
                into_mut_expand_element(scope, elem)
            }
        }

        impl IntoMut for $primitive {
            fn into_mut(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl Float for $primitive {
            const DIGITS: u32 = $primitive::DIGITS;
            const EPSILON: Self = $primitive::EPSILON;
            const INFINITY: Self = $primitive::INFINITY;
            const MANTISSA_DIGITS: u32 = $primitive::MANTISSA_DIGITS;
            const MAX_10_EXP: i32 = $primitive::MAX_10_EXP;
            const MAX_EXP: i32 = $primitive::MAX_EXP;
            const MIN_10_EXP: i32 = $primitive::MIN_10_EXP;
            const MIN_EXP: i32 = $primitive::MIN_EXP;
            const MIN_POSITIVE: Self = $primitive::MIN_POSITIVE;
            const NAN: Self = $primitive::NAN;
            const NEG_INFINITY: Self = $primitive::NEG_INFINITY;
            const RADIX: u32 = $primitive::RADIX;

            fn new(val: f32) -> Self {
                $new(val)
            }
        }

        impl LaunchArgExpand for $primitive {
            type CompilationArg = ();

            fn expand(
                _: &Self::CompilationArg,
                builder: &mut KernelBuilder,
            ) -> ExpandElementTyped<Self> {
                builder.scalar($primitive::as_elem(&builder.scope)).into()
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
