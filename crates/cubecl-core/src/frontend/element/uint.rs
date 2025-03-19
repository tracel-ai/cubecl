use cubecl_ir::{ExpandElement, Scope, UIntKind};

use crate::Runtime;
use crate::frontend::{CubePrimitive, CubeType, Numeric};
use crate::ir::Elem;
use crate::prelude::{KernelBuilder, KernelLauncher};

use super::{
    ExpandElementBaseInit, ExpandElementTyped, Init, Int, IntoRuntime, LaunchArgExpand,
    ScalarArgSettings, init_expand_element,
};

macro_rules! declare_uint {
    ($primitive:ident, $kind:ident) => {
        impl CubeType for $primitive {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl ExpandElementBaseInit for $primitive {
            fn init_elem(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
                init_expand_element(scope, elem)
            }
        }

        impl CubePrimitive for $primitive {
            fn as_elem_native() -> Option<Elem> {
                Some(Elem::UInt(UIntKind::$kind))
            }
        }

        impl Init for $primitive {
            fn init(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
                let expand: ExpandElementTyped<Self> = self.into();
                Init::init(expand, scope)
            }
        }

        impl LaunchArgExpand for $primitive {
            type CompilationArg = ();

            fn expand(
                _: &Self::CompilationArg,
                builder: &mut KernelBuilder,
            ) -> ExpandElementTyped<Self> {
                builder.scalar($primitive::as_elem(&builder.context)).into()
            }
        }

        impl Numeric for $primitive {
            fn min_value() -> Self {
                $primitive::MIN
            }
            fn max_value() -> Self {
                $primitive::MAX
            }
        }

        impl Int for $primitive {
            const BITS: u32 = $primitive::BITS;

            fn new(val: i64) -> Self {
                val as $primitive
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
