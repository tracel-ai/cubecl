use cubecl_ir::{ConstantValue, ElemType};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    r#type::TypeHandle,
};

use crate::frontend::{CubeType, Numeric};
use crate::ir::{IntKind, Scope};
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
    + CubeBitOr
    + CubeBitAnd
    + CubeBitXor
    + CubeShl
    + CubeShr
    + CubeNot
    + CubeBitOrAssign
    + CubeBitAndAssign
    + CubeBitXorAssign
    + CubeShlAssign
    + CubeShrAssign
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
    fn __expand_new(scope: &Scope, val: i64) -> <Self as CubeType>::ExpandType {
        __expand_new(scope, val)
    }

    fn is_signed(scope: &Scope) -> bool {
        Self::elem_type(scope).is_signed_int()
    }
}

macro_rules! impl_int {
    ($type: ident, $kind: ident) => {
        impl CubeType for $type {
            type ExpandType = NativeExpand<Self>;
        }

        impl CubeDebug for $type {}
        impl Scalar for $type {
            fn elem_type_native() -> ElemType {
                IntKind::$kind.into()
            }
        }
        impl CubePrimitive for $type {
            type Scalar = Self;
            type Size = Const<1>;
            type WithScalar<S: Scalar> = S;

            fn __expand_as_type(scope: &Scope) -> TypeHandle {
                let width = IntKind::$kind.size_bits() as u32;
                IntegerType::get(scope.ctx(), width, Signedness::Signed).into()
            }

            fn from_const_value(value: ConstantValue) -> Self {
                let ConstantValue::Int(value) = value else {
                    unreachable!()
                };
                value as $type
            }
        }

        impl IntoRuntime for $type {
            fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
                self.into()
            }
        }

        impl IntoExpand for $type {
            type Expand = NativeExpand<$type>;

            fn into_expand(self, _: &Scope) -> Self::Expand {
                self.into()
            }
        }

        impl IntoMut for $type {
            fn into_mut(self, _scope: &Scope) -> Self {
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

        impl_scalar_launch!($type);
    };
}

impl_int!(i8, I8);
impl_int!(i16, I16);
impl_int!(i32, I32);
impl_int!(i64, I64);
