use crate::prelude::{KernelBuilder, KernelLauncher};
use crate::Runtime;
use crate::{
    frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, Numeric},
    ir::UIntKind,
};
use crate::{ir::Elem, unexpanded};

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, Init, Int, IntoRuntime,
    LaunchArgExpand, ScalarArgSettings, Vectorized,
};

macro_rules! declare_uint {
    ($primitive:ident, $kind:ident) => {
        impl CubeType for $primitive {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl ExpandElementBaseInit for $primitive {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl CubePrimitive for $primitive {
            fn as_elem() -> Elem {
                Elem::UInt(UIntKind::$kind)
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(
                self,
                context: &mut CubeContext,
            ) -> ExpandElementTyped<Self> {
                let expand: ExpandElementTyped<Self> = self.into();
                Init::init(expand, context)
            }
        }

        impl LaunchArgExpand for $primitive {
            type CompilationArg = ();

            fn expand(
                _: &Self::CompilationArg,
                builder: &mut KernelBuilder,
            ) -> ExpandElementTyped<Self> {
                builder.scalar($primitive::as_elem()).into()
            }
        }

        impl Numeric for $primitive {
            const MAX: Self = $primitive::MAX;
            const MIN: Self = $primitive::MIN;
        }

        impl Int for $primitive {
            const BITS: u32 = $primitive::BITS;

            fn new(val: i64) -> Self {
                val as $primitive
            }

            fn vectorized(val: i64, _vectorization: u32) -> Self {
                Self::new(val)
            }
        }

        impl Vectorized for $primitive {
            fn vectorization_factor(&self) -> u32 {
                1
            }

            fn vectorize(self, _factor: u32) -> Self {
                unexpanded!()
            }
        }
    };
}

declare_uint!(u8, U8);
declare_uint!(u16, U16);
declare_uint!(u32, U32);
declare_uint!(u64, U64);

impl ScalarArgSettings for u8 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u8(*self);
    }
}

impl ScalarArgSettings for u16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u16(*self);
    }
}

impl ScalarArgSettings for u32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(*self);
    }
}

impl ScalarArgSettings for u64 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u64(*self);
    }
}
