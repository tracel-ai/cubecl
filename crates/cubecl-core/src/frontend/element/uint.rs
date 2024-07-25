use crate::frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, Vectorization};
use crate::prelude::{KernelBuilder, KernelLauncher};
use crate::{frontend::Comptime, Runtime};

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, LaunchArgExpand,
    ScalarArgSettings, Vectorized, __expand_new, __expand_vectorized,
};

#[derive(Clone, Copy, Debug)]
/// An unsigned int.
/// Preferred for indexing operations
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

impl CubeType for UInt {
    type ExpandType = ExpandElementTyped<Self>;
}

impl ExpandElementBaseInit for UInt {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl CubePrimitive for UInt {
    type Primitive = u32;
    fn as_elem() -> Elem {
        Elem::UInt
    }
}

impl LaunchArgExpand for UInt {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Self> {
        assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
        builder.scalar(UInt::as_elem()).into()
    }
}

impl ScalarArgSettings for u32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(*self);
    }
}

impl Numeric for UInt {}

impl UInt {
    pub const fn new(val: u32) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    pub fn vectorized(val: u32, vectorization: UInt) -> Self {
        if vectorization.val == 1 {
            Self::new(val)
        } else {
            Self {
                val,
                vectorization: vectorization.val as u8,
            }
        }
    }
    pub fn __expand_new(
        context: &mut CubeContext,
        val: <Self as CubeType>::ExpandType,
    ) -> <Self as CubeType>::ExpandType {
        __expand_new(context, val, Self::as_elem())
    }

    pub fn __expand_vectorized(
        context: &mut CubeContext,
        val: <Self as CubeType>::ExpandType,
        vectorization: UInt,
    ) -> <Self as CubeType>::ExpandType {
        __expand_vectorized(context, val, vectorization, Self::as_elem())
    }
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value)
    }
}

impl From<Comptime<u32>> for UInt {
    fn from(value: Comptime<u32>) -> Self {
        UInt::new(value.inner)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32)
    }
}

impl From<i32> for UInt {
    fn from(value: i32) -> Self {
        UInt::new(value as u32)
    }
}

impl Vectorized for UInt {
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
