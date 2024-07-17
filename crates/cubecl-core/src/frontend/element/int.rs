use crate::compute::{KernelBuilder, KernelLauncher};
use crate::frontend::{
    ComptimeType, CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementBaseInit,
    ExpandElementTyped, Numeric,
};
use crate::ir::{ConstantScalarValue, Elem, IntKind, Variable, Vectorization};
use crate::Runtime;

use super::{
    init_expand_element, LaunchArgExpand, ScalarArgSettings, UInt, Vectorized, __expand_new,
    __expand_vectorized,
};

/// Signed integer. Used as input in int kernels
pub trait Int: Numeric + std::ops::Rem<Output = Self> {
    fn new(val: i64) -> Self;
    fn vectorized(val: i64, vectorization: UInt) -> Self;
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

macro_rules! impl_int {
    ($type:ident, $primitive:ty) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: $primitive,
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl CubePrimitive for $type {
            fn as_elem() -> Elem {
                Elem::Int(IntKind::$type)
            }
        }

        impl ComptimeType for $type {
            fn into_expand(self) -> Self::ExpandType {
                let elem = Self::as_elem();
                let value = match elem {
                    Elem::Int(kind) => ConstantScalarValue::Int(self.val as i64, kind),
                    Elem::UInt => ConstantScalarValue::UInt(self.val as u64),
                    _ => panic!("Wrong elem type"),
                };

                ExpandElementTyped::new(ExpandElement::Plain(Variable::ConstantScalar(value)))
            }
        }

        impl Numeric for $type {
            type Primitive = $primitive;
        }

        impl ExpandElementBaseInit for $type {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl Int for $type {
            fn new(val: i64) -> Self {
                Self {
                    val: val as $primitive,
                    vectorization: 1,
                }
            }

            fn vectorized(val: i64, vectorization: UInt) -> Self {
                if vectorization.val == 1 {
                    Self::new(val)
                } else {
                    Self {
                        val: val as $primitive,
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

impl_int!(I32, i32);
impl_int!(I64, i64);

impl From<i64> for I64 {
    fn from(value: i64) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<i32> for I32 {
    fn from(value: i32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

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
