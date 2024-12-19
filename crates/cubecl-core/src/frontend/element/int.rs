use crate::compute::{KernelBuilder, KernelLauncher};
use crate::frontend::{
    CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementBaseInit, ExpandElementTyped,
    Numeric,
};
use crate::ir::{Elem, IntKind};
use crate::prelude::TypeMap;
use crate::Runtime;

use super::{
    init_expand_element, Init, IntExpand, IntoRuntime, LaunchArgExpand, ScalarArgSettings,
    __expand_new,
};

/// Signed or unsigned integer. Used as input in int kernels
pub trait Int:
    Numeric
    + std::ops::Rem<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::BitOr<Output = Self>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitXor<Output = Self>
    + core::ops::Shl<Output = Self>
    + core::ops::Shr<Output = Self>
    + std::ops::RemAssign
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::BitOrAssign
    + std::ops::BitAndAssign
    + std::ops::BitXorAssign
    + std::ops::ShlAssign<u32>
    + std::ops::ShrAssign<u32>
    + std::cmp::PartialOrd
    + std::cmp::PartialEq
{
    const BITS: u32;

    fn new(val: i64) -> Self;
    fn __expand_new(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
        __expand_new(context, val)
    }
}

macro_rules! impl_int {
    ($type:ident, $kind:ident) => {
        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl<const POS: u8> TypeMap<POS> for $type {
            type ExpandGeneric = IntExpand<POS>;

            fn register(context: &mut CubeContext) {
                let elem = Self::as_elem(context);
                context.register_type::<Self::ExpandGeneric>(elem);
            }
        }

        impl CubePrimitive for $type {
            fn as_elem_native() -> Option<Elem> {
                Some(Elem::Int(IntKind::$kind))
            }
        }

        impl IntoRuntime for $type {
            fn __expand_runtime_method(
                self,
                context: &mut CubeContext,
            ) -> ExpandElementTyped<Self> {
                let expand: ExpandElementTyped<Self> = self.into();
                Init::init(expand, context)
            }
        }

        impl Numeric for $type {
            const MAX: Self = $type::MAX;
            const MIN: Self = $type::MIN;
        }

        impl ExpandElementBaseInit for $type {
            fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
                init_expand_element(context, elem)
            }
        }

        impl Int for $type {
            const BITS: u32 = $type::BITS;

            fn new(val: i64) -> Self {
                val as $type
            }
        }

        impl LaunchArgExpand for $type {
            type CompilationArg = ();

            fn expand(
                _: &Self::CompilationArg,
                builder: &mut KernelBuilder,
            ) -> ExpandElementTyped<Self> {
                builder.scalar($type::as_elem(&builder.context)).into()
            }
        }
    };
}

impl_int!(i8, I8);
impl_int!(i16, I16);
impl_int!(i32, I32);
impl_int!(i64, I64);

impl ScalarArgSettings for i8 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i8(*self);
    }
}

impl ScalarArgSettings for i16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i16(*self);
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
