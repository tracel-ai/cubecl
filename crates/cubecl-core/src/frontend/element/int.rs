use cubecl_ir::ExpandElement;

use crate::Runtime;
use crate::frontend::{CubeType, Numeric};
use crate::ir::{Elem, IntKind, Scope};
use crate::prelude::BitwiseNot;
use crate::prelude::{FindFirstSet, LeadingZeros};
use crate::{
    compute::{KernelBuilder, KernelLauncher},
    prelude::{CountOnes, ReverseBits},
};

use super::{
    __expand_new, CubePrimitive, ExpandElementIntoMut, ExpandElementTyped, IntoMut, IntoRuntime,
    LaunchArgExpand, ScalarArgSettings, into_mut_expand_element, into_runtime_expand_element,
};

mod typemap;
pub use typemap::*;

/// Signed or unsigned integer. Used as input in int kernels
pub trait Int:
    Numeric
    + CountOnes
    + ReverseBits
    + BitwiseNot
    + LeadingZeros
    + FindFirstSet
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
    + core::ops::Not<Output = Self>
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
    fn __expand_new(scope: &mut Scope, val: i64) -> <Self as CubeType>::ExpandType {
        __expand_new(scope, val)
    }
}

macro_rules! impl_int {
    ($type:ident, $kind:ident) => {
        impl CubeType for $type {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl CubePrimitive for $type {
            fn as_elem_native() -> Option<Elem> {
                Some(Elem::Int(IntKind::$kind))
            }
        }

        impl IntoRuntime for $type {
            fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
                let elem: ExpandElementTyped<Self> = self.into();
                into_runtime_expand_element(scope, elem).into()
            }
        }

        impl IntoMut for $type {
            fn into_mut(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl Numeric for $type {
            fn min_value() -> Self {
                $type::MIN
            }
            fn max_value() -> Self {
                $type::MAX
            }
        }

        impl ExpandElementIntoMut for $type {
            fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
                into_mut_expand_element(scope, elem)
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
                builder.scalar($type::as_elem(&builder.scope)).into()
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
