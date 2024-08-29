use cubecl_macros_2::expand_impl;

use crate::prelude::{KernelBuilder, KernelLauncher};
use crate::{frontend::Comptime, Runtime};
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, Numeric},
    new_ir::Expand,
};
use crate::{
    ir::{Elem, Vectorization},
    new_ir::Expr,
};

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, LaunchArgExpand,
    ScalarArgSettings, Vectorized, __expand_new, __expand_vectorized,
};

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Copy, Hash)]
/// An unsigned int.
/// Preferred for indexing operations
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

pub struct UIntExpand<Inner: Expr<Output = UInt>>(Inner);

impl Expand for UInt {
    type Expanded<Inner: Expr<Output = Self>> = UIntExpand<Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        UIntExpand(inner)
    }
}

impl core::fmt::Debug for UInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.vectorization == 1 {
            f.write_fmt(format_args!("{}", self.val))
        } else {
            f.write_fmt(format_args!("{}-{}", self.val, self.vectorization))
        }
    }
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

impl Numeric for UInt {
    type Primitive = u32;
}

#[expand_impl]
impl UInt {
    pub const fn new(val: u32) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    #[expanded]
    pub const fn new(val: u32) -> UInt {
        UInt {
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

    #[expanded]
    pub fn vectorized(val: u32, vectorization: UInt) -> UInt {
        if vectorization.val == 1 {
            UInt::new(val)
        } else {
            UInt {
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
