use core::cmp::Ordering;
use cubecl_common::*;
use half::{bf16, f16};

use cubecl_ir::{ExpandValue, dialect::cmp::*};

use crate as cubecl;
use crate::frontend::NativeExpand;
use crate::ir::Scope;
use crate::prelude::*;

// NOTE: Unary comparison tests are in the unary module

pub trait CubePartialEq:
    PartialEq
    + CubePrimitive<Scalar: PartialEqNativeExpand>
    + CubeType<ExpandType: PartialEqExpand>
    + Sized
    + IntoExpand<Expand = <Self as CubeType>::ExpandType>
{
    fn __expand_eq_method(&self, scope: &Scope, rhs: &NativeExpand<Self>) -> NativeExpand<bool> {
        let this = (*self).into_expand(scope);
        Self::__expand_eq(scope, &this, rhs)
    }
    fn __expand_ne_method(&self, scope: &Scope, rhs: &NativeExpand<Self>) -> NativeExpand<bool> {
        let this = (*self).into_expand(scope);
        Self::__expand_ne(scope, &this, rhs)
    }

    fn __expand_eq(
        scope: &Scope,
        lhs: &NativeExpand<Self>,
        rhs: &NativeExpand<Self>,
    ) -> NativeExpand<bool> {
        lhs.__expand_eq_method(scope, rhs)
    }
    fn __expand_ne(
        scope: &Scope,
        lhs: &NativeExpand<Self>,
        rhs: &NativeExpand<Self>,
    ) -> NativeExpand<bool> {
        lhs.__expand_ne_method(scope, rhs)
    }
}
pub trait PartialEqExpand {
    fn __expand_eq_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_ne_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool>;
}
pub trait PartialEqNativeExpand {
    fn __expand_native_eq(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
    fn __expand_native_ne(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
}

impl<
    T: PartialEq
        + CubePrimitive<Scalar: PartialEqNativeExpand>
        + IntoExpand<Expand = <Self as CubeType>::ExpandType>,
> CubePartialEq for T
{
}

impl<T: CubePartialEq> PartialEqExpand for NativeExpand<T> {
    fn __expand_eq_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        T::Scalar::__expand_native_eq(scope, this.expand, rhs.expand).into()
    }
    fn __expand_ne_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        T::Scalar::__expand_native_ne(scope, this.expand, rhs.expand).into()
    }
}

macro_rules! impl_partial_eq {
    ($($ty: ty),*; $eq: ty, $ne: ty) => {
        $(impl PartialEqNativeExpand for $ty {
            fn __expand_native_eq(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$eq>::new)
            }
            fn __expand_native_ne(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$ne>::new)
            }
        })*
    };
}

impl_partial_eq!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; IEqualOp, INotEqualOp);
impl_partial_eq!(f16, bf16, f32, flex32, tf32, f64; FEqualOp, FNotEqualOp);
impl_partial_eq!(e2m1, e2m1x2, e3m2, e2m3, e4m3, e5m2, ue8m0; FEqualOp, FNotEqualOp);
impl_partial_eq!(bool; BoolEqualOp, BoolNotEqualOp);

#[derive_expand(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants, no_constructors)]
pub enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

fn ordering_disc(name: &'static str) -> NativeExpand<i32> {
    OrderingExpand::discriminant_of(name).into()
}

#[allow(non_snake_case)]
pub trait CubeOrdering {
    fn Less() -> Ordering {
        Ordering::Less
    }
    fn Equal() -> Ordering {
        Ordering::Equal
    }
    fn Greater() -> Ordering {
        Ordering::Greater
    }
    fn __expand_Less(_scope: &Scope) -> OrderingExpand {
        OrderingExpand {
            discriminant: ordering_disc("Less"),
            value: (),
        }
    }
    fn __expand_Equal(_scope: &Scope) -> OrderingExpand {
        OrderingExpand {
            discriminant: ordering_disc("Equal"),
            value: (),
        }
    }
    fn __expand_Greater(_scope: &Scope) -> OrderingExpand {
        OrderingExpand {
            discriminant: ordering_disc("Greater"),
            value: (),
        }
    }
}

impl CubeOrdering for Ordering {}

pub trait CubeOrd:
    Ord
    + CubePartialOrd
    + CubeType<ExpandType: OrdExpand>
    + CubePrimitive<Scalar: OrdNativeExpand>
    + Sized
    + IntoExpand<Expand = <Self as CubeType>::ExpandType>
{
    fn __expand_min_method(self, scope: &Scope, rhs: Self::ExpandType) -> Self::ExpandType {
        let this = self.into_expand(scope);
        Self::__expand_min(scope, this, rhs)
    }
    fn __expand_max_method(self, scope: &Scope, rhs: Self::ExpandType) -> Self::ExpandType {
        let this = self.into_expand(scope);
        Self::__expand_max(scope, this, rhs)
    }
    fn __expand_clamp_method(
        self,
        scope: &Scope,
        min: Self::ExpandType,
        max: Self::ExpandType,
    ) -> Self::ExpandType {
        let this = self.into_expand(scope);
        Self::__expand_clamp(scope, this, min, max)
    }

    fn __expand_cmp(
        scope: &Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> OrderingExpand {
        lhs.__expand_cmp_method(scope, rhs)
    }

    fn __expand_min(
        scope: &Scope,
        lhs: Self::ExpandType,
        rhs: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_min_method(scope, rhs)
    }

    fn __expand_max(
        scope: &Scope,
        lhs: Self::ExpandType,
        rhs: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_max_method(scope, rhs)
    }

    fn __expand_clamp(
        scope: &Scope,
        lhs: Self::ExpandType,
        min: Self::ExpandType,
        max: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_clamp_method(scope, min, max)
    }
}
pub trait OrdExpand {
    fn __expand_cmp_method(&self, scope: &Scope, rhs: &Self) -> OrderingExpand;
    fn __expand_min_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_max_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_clamp_method(self, scope: &Scope, min: Self, max: Self) -> Self;
}
pub trait OrdNativeExpand {
    fn __expand_native_min(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
    fn __expand_native_max(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
    fn __expand_native_clamp(
        scope: &Scope,
        input: ExpandValue,
        min: ExpandValue,
        max: ExpandValue,
    ) -> ExpandValue;
}

macro_rules! impl_ord {
    ($($ty: ty),*; $min: ty, $max: ty, $clamp: ty) => {
        $(impl OrdNativeExpand for $ty {
            fn __expand_native_min(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$min>::new)
            }
            fn __expand_native_max(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$max>::new)
            }
            fn __expand_native_clamp(
                scope: &Scope,
                input: ExpandValue,
                min: ExpandValue,
                max: ExpandValue,
            ) -> ExpandValue {
                let input = input.read_value(scope);
                let min = min.read_value(scope);
                let max = max.read_value(scope);
                let op = <$clamp>::new(scope.ctx_mut(), input, min, max);
                scope.register_with_result(&op).into()
            }
        })*
    };
}

impl_ord!(i8, i16, i32, i64, isize; SMinOp, SMaxOp, SClampOp);
impl_ord!(u8, u16, u32, u64, usize; UMinOp, UMaxOp, UClampOp);
impl_ord!(f16, bf16, f32, flex32, tf32, f64; FMinOp, FMaxOp, FClampOp);

impl<
    T: Ord
        + CubePartialOrd
        + CubePrimitive<Scalar: OrdNativeExpand>
        + IntoExpand<Expand = <Self as CubeType>::ExpandType>,
> CubeOrd for T
{
}
impl<T: CubeOrd> OrdExpand for NativeExpand<T> {
    fn __expand_cmp_method(&self, scope: &Scope, rhs: &Self) -> OrderingExpand {
        let lhs_lt_rhs = self.__expand_lt_method(scope, rhs);
        let lhs_gt_rhs = self.__expand_gt_method(scope, rhs);
        let less = ordering_disc("Less");
        let equal = ordering_disc("Equal");
        let greater = ordering_disc("Greater");
        let eq_or_gt = select::expand(scope, lhs_gt_rhs, greater, equal);
        let discriminant = select::expand(scope, lhs_lt_rhs, less, eq_or_gt);
        OrderingExpand {
            discriminant,
            value: (),
        }
    }
    fn __expand_min_method(self, scope: &Scope, rhs: Self) -> Self {
        min::expand(scope, self, rhs)
    }
    fn __expand_max_method(self, scope: &Scope, rhs: Self) -> Self {
        max::expand(scope, self, rhs)
    }
    fn __expand_clamp_method(self, scope: &Scope, min: Self, max: Self) -> Self {
        clamp::expand(scope, self, min, max)
    }
}

pub trait CubePartialOrd:
    PartialOrd
    + CubeType<ExpandType: PartialOrdExpand>
    + CubePrimitive<Scalar: PartialOrdScalarExpand + OrdNativeExpand>
    + Sized
{
    fn __expand_partial_cmp(
        scope: &Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> OptionExpand<Ordering> {
        lhs.__expand_partial_cmp_method(scope, rhs)
    }

    fn __expand_lt(
        scope: &Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_lt_method(scope, rhs)
    }

    fn __expand_le(
        scope: &Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_le_method(scope, rhs)
    }

    fn __expand_gt(
        scope: &Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_gt_method(scope, rhs)
    }

    fn __expand_ge(
        scope: &Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_ge_method(scope, rhs)
    }
}

pub trait PartialOrdExpand {
    fn __expand_partial_cmp_method(&self, scope: &Scope, rhs: &Self) -> OptionExpand<Ordering>;
    fn __expand_lt_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_le_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_gt_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_ge_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool>;
}

pub trait PartialOrdScalarExpand {
    fn __expand_native_lt(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
    fn __expand_native_le(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
    fn __expand_native_gt(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
    fn __expand_native_ge(scope: &Scope, lhs: ExpandValue, rhs: ExpandValue) -> ExpandValue;
}

macro_rules! impl_partial_ord {
    ($($ty: ty),*; $lt: ty, $le: ty, $gt: ty, $ge: ty) => {
        $(impl PartialOrdScalarExpand for $ty {
            fn __expand_native_lt(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$lt>::new)
            }
            fn __expand_native_le(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$le>::new)
            }
            fn __expand_native_gt(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$gt>::new)
            }
            fn __expand_native_ge(
                scope: &Scope,
                lhs: ExpandValue,
                rhs: ExpandValue,
            ) -> ExpandValue {
                binary_expand(scope, lhs, rhs, <$ge>::new)
            }
        })*
    };
}

impl_partial_ord!(i8, i16, i32, i64, isize; SLessThanOp, SLessThanOrEqualOp, SGreaterThanOp, SGreaterThanOrEqualOp);
impl_partial_ord!(u8, u16, u32, u64, usize; ULessThanOp, ULessThanOrEqualOp, UGreaterThanOp, UGreaterThanOrEqualOp);
impl_partial_ord!(f16, bf16, f32, flex32, tf32, f64; FLessThanOp, FLessThanOrEqualOp, FGreaterThanOp, FGreaterThanOrEqualOp);

impl<T: PartialOrd + CubePrimitive<Scalar: PartialOrdScalarExpand + OrdNativeExpand>> CubePartialOrd
    for T
{
}
impl<T: CubePartialOrd> PartialOrdExpand for NativeExpand<T> {
    fn __expand_partial_cmp_method(&self, scope: &Scope, rhs: &Self) -> OptionExpand<Ordering> {
        let lhs_lt_rhs = self.__expand_lt_method(scope, rhs);
        let lhs_gt_rhs = self.__expand_gt_method(scope, rhs);
        let less = ordering_disc("Less");
        let equal = ordering_disc("Equal");
        let greater = ordering_disc("Greater");
        let eq_or_gt = select::expand(scope, lhs_gt_rhs, greater, equal);
        let discriminant = select::expand(scope, lhs_lt_rhs, less, eq_or_gt);
        Option::__expand_new_Some(
            scope,
            OrderingExpand {
                discriminant,
                value: (),
            },
        )
    }
    fn __expand_lt_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        T::Scalar::__expand_native_lt(scope, this.into(), rhs.into()).into()
    }
    fn __expand_le_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        T::Scalar::__expand_native_le(scope, this.into(), rhs.into()).into()
    }
    fn __expand_gt_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        T::Scalar::__expand_native_gt(scope, this.into(), rhs.into()).into()
    }
    fn __expand_ge_method(&self, scope: &Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        T::Scalar::__expand_native_ge(scope, this.into(), rhs.into()).into()
    }
}
