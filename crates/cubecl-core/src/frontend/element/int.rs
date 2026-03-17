use cubecl_ir::{ConstantValue, Type};

use crate::frontend::{CubeType, Numeric};
use crate::ir::{ElemType, IntKind, Scope};
use crate::prelude::*;

use super::{__expand_new, CubePrimitive, IntoMut, IntoRuntime, NativeAssign, NativeExpand};

/// Signed or unsigned integer. Used as input in int kernels
pub trait Int:
    Numeric
    + CubeNot
    + CountOnes
    + ReverseBits
    + LeadingZeros
    + TrailingZeros
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
            type ExpandType = NativeExpand<Self>;
        }

        impl Scalar for $type {}
        impl CubePrimitive for $type {
            type Scalar = Self;
            type Size = Const<1>;
            type WithScalar<S: Scalar> = S;

            fn as_type_native() -> Option<Type> {
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
            fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
                self.into()
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

        impl NativeAssign for $type {}

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
