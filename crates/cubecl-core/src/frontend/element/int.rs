use cubecl_ir::{ConstantValue, ExpandElement, StorageType};

use crate::ir::{ElemType, IntKind, Scope};
use crate::prelude::{CountOnes, ReverseBits};
use crate::prelude::{FindFirstSet, LeadingZeros, SaturatingAdd, SaturatingSub};
use crate::{
    frontend::{CubeType, Numeric},
    prelude::CubeNot,
};

use super::{
    __expand_new, CubePrimitive, ExpandElementIntoMut, ExpandElementTyped, IntoMut, IntoRuntime,
    into_mut_expand_element, into_runtime_expand_element,
};

mod typemap;
pub use typemap::*;

/// Signed or unsigned integer. Used as input in int kernels
pub trait Int:
    Numeric
    + CubeNot
    + CountOnes
    + ReverseBits
    + LeadingZeros
    + FindFirstSet
    + SaturatingAdd
    + SaturatingSub
    + core::ops::BitOr<Output = Self>
    + core::ops::BitAnd<Output = Self>
    + core::ops::BitXor<Output = Self>
    + core::ops::Shl<Output = Self>
    + core::ops::Shr<Output = Self>
    + core::ops::Not<Output = Self>
    + core::ops::BitOrAssign
    + core::ops::BitAndAssign
    + core::ops::BitXorAssign
    + core::ops::ShlAssign<u32>
    + core::ops::ShrAssign<u32>
    + core::hash::Hash
    + core::cmp::PartialOrd
    + core::cmp::Ord
    + core::cmp::PartialEq
    + core::cmp::Eq
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
            fn as_type_native() -> Option<StorageType> {
                Some(ElemType::Int(IntKind::$kind).into())
            }

            fn from_const_value(value: ConstantValue) -> Self {
                let ConstantValue::Int(value) = value else {
                    unreachable!()
                };
                value as $type
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
    };
}

impl_int!(i8, I8);
impl_int!(i16, I16);
impl_int!(i32, I32);
impl_int!(i64, I64);
