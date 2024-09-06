use crate::compute::{KernelBuilder, KernelLauncher};
use crate::frontend::{
    CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementBaseInit, ExpandElementTyped,
    Numeric,
};
use crate::ir::{Elem, IntKind, Vectorization};
use crate::Runtime;

use super::{
    init_expand_element, LaunchArgExpand, ScalarArgSettings, __expand_new, __expand_vectorized,
};

/// Signed integer. Used as input in int kernels
pub trait Int:
    Numeric
    + std::ops::Rem<Output = Self>
    + From<i32>
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
    fn new(val: i64) -> Self;
    fn vectorized(val: i64, vectorization: u32) -> Self;
    fn __expand_new(
        context: &mut CubeContext,
        val: Self::ExpandType,
    ) -> <Self as CubeType>::ExpandType {
        __expand_new(context, val, Self::as_elem())
    }
    fn __expand_vectorized(
        context: &mut CubeContext,
        val: Self::ExpandType,
        vectorization: u32,
    ) -> <Self as CubeType>::ExpandType {
        __expand_vectorized(context, val, vectorization, Self::as_elem())
    }
}

macro_rules! impl_int {
    ($type:ident, $kind:ident) => {
        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl CubePrimitive for $type {
            fn as_elem() -> Elem {
                Elem::Int(IntKind::$kind)
            }
        }

        impl Numeric for $type {}

        impl ExpandElementBaseInit for $type {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl Int for $type {
            fn new(val: i64) -> Self {
                val as $type
            }

            fn vectorized(val: i64, _vectorization: u32) -> Self {
                Self::new(val)
            }
        }

        impl LaunchArgExpand for $type {
            fn expand(
                builder: &mut KernelBuilder,
                vectorization: Vectorization,
            ) -> ExpandElementTyped<Self> {
                assert_eq!(vectorization, None, "Attempted to vectorize a scalar");
                builder.scalar($type::as_elem()).into()
            }
        }
    };
}

impl_int!(i32, I32);
impl_int!(i64, I64);

impl ScalarArgSettings for i32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i32(*self);
    }
}

impl ScalarArgSettings for i64 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i64(*self);
    }
}
