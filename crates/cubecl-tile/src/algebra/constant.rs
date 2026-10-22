//! Integer arithmetic on kernel values that folds constants at expand time.

use cubecl::ir::{
    ConstantValue, ExpandValue, Scope,
    interfaces::{ScalarType, TypedExt},
    try_cast_ty,
};
use cubecl::prelude::*;
use cubecl::unexpanded;

/// Arithmetic on an integer kernel value that a constant survives.
pub(crate) trait Integer: Sized {
    /// `self + rhs`; `x + 0` passes through.
    fn plus(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self - rhs`; `x - 0` passes through.
    fn minus(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self * rhs`; `x * 1` passes through, `x * 0` is `0`.
    fn times(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self / rhs`; `x / 1` passes through.
    fn divided_by(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// `self % rhs`; `x % 1` is `0`.
    fn remainder(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The smaller of the two; two constants fold.
    fn min_with(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The larger of the two; two constants fold.
    fn max_with(self, _rhs: Self) -> Self {
        unexpanded!()
    }
    /// The value re-typed to `To`, a constant staying constant.
    fn cast<To: Int>(self) -> To {
        unexpanded!()
    }
    /// The comptime constant this value holds, if any.
    fn constant(self) -> Option<u64> {
        unexpanded!()
    }
}

impl Integer for u32 {}
impl Integer for usize {}
impl Integer for i32 {}

/// Constant-keeping sums over the elements at comptime `picks`.
pub(crate) trait IntegerSeq<C: Int>: Sized {
    /// Sum of the picked elements (empty picks fold to `0`).
    fn sum(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }
}

impl<C: Int + Integer> IntegerSeq<C> for Sequence<C> {}

/// The constant a non-negative integer expand element holds, if any.
pub(crate) fn constant<C: Int>(e: &NativeExpand<C>) -> Option<u64> {
    match e.expand.as_const() {
        Some(ConstantValue::UInt(v)) => Some(v),
        Some(ConstantValue::Int(v)) if v >= 0 => Some(v as u64),
        _ => None,
    }
}

fn constant_like<C: Int>(scope: &Scope, v: u64, e: &NativeExpand<C>) -> NativeExpand<C> {
    let ty = match e.expand {
        ExpandValue::Value(val) => {
            let ctx = scope.ctx();
            let scalar = val.try_get_scalar_elem_ty(ctx).unwrap().deref(ctx);
            try_cast_ty!(scalar, ctx, dyn ScalarType).elem_type(ctx)
        }
        ExpandValue::Constant { ty, .. } => ty,
    };
    ExpandValue::constant(v.into(), ty).into()
}

pub(crate) fn fold_add<C: Int>(
    scope: &Scope,
    lhs: NativeExpand<C>,
    rhs: NativeExpand<C>,
) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a + b, &lhs),
        (Some(0), None) => rhs,
        (None, Some(0)) => lhs,
        _ => AddExpand::__expand_add_method(lhs, scope, rhs),
    }
}

fn fold_sub<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if a >= b => constant_like(scope, a - b, &lhs),
        (None, Some(0)) => lhs,
        _ => SubExpand::__expand_sub_method(lhs, scope, rhs),
    }
}

pub(crate) fn fold_mul<C: Int>(
    scope: &Scope,
    lhs: NativeExpand<C>,
    rhs: NativeExpand<C>,
) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a * b, &lhs),
        (Some(0), None) | (None, Some(0)) => constant_like(scope, 0, &lhs),
        (Some(1), None) => rhs,
        (None, Some(1)) => lhs,
        _ => MulExpand::__expand_mul_method(lhs, scope, rhs),
    }
}

fn fold_div<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if b != 0 => constant_like(scope, a / b, &lhs),
        (None, Some(1)) => lhs,
        // A divisor here is an extent, never 0.
        (Some(0), None) => constant_like(scope, 0, &lhs),
        _ => DivExpand::__expand_div_method(lhs, scope, rhs),
    }
}

fn fold_rem<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) if b != 0 => constant_like(scope, a % b, &lhs),
        (None, Some(1)) | (Some(0), None) => constant_like(scope, 0, &lhs),
        _ => RemExpand::__expand_rem_method(lhs, scope, rhs),
    }
}

fn fold_min<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a.min(b), &lhs),
        _ => C::__expand_min(scope, lhs, rhs),
    }
}

fn fold_max<C: Int>(scope: &Scope, lhs: NativeExpand<C>, rhs: NativeExpand<C>) -> NativeExpand<C> {
    match (constant(&lhs), constant(&rhs)) {
        (Some(a), Some(b)) => constant_like(scope, a.max(b), &lhs),
        _ => C::__expand_max(scope, lhs, rhs),
    }
}

/// Expand twin of [`Integer`]; blanket on integer expand elements.
pub(crate) trait IntegerExpand<C: Int>: Sized {
    fn __expand_plus_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_minus_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_times_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_divided_by_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_remainder_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_min_with_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_max_with_method(self, scope: &Scope, rhs: Self) -> Self;
    fn __expand_cast_method<To: Int>(self, scope: &Scope) -> NativeExpand<To>;
    fn __expand_constant_method(self, scope: &Scope) -> Option<u64>;
}

impl<C: Int> IntegerExpand<C> for NativeExpand<C> {
    fn __expand_plus_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_add(scope, self, rhs)
    }
    fn __expand_minus_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_sub(scope, self, rhs)
    }
    fn __expand_times_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_mul(scope, self, rhs)
    }
    fn __expand_divided_by_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_div(scope, self, rhs)
    }
    fn __expand_remainder_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_rem(scope, self, rhs)
    }
    fn __expand_min_with_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_min(scope, self, rhs)
    }
    fn __expand_max_with_method(self, scope: &Scope, rhs: Self) -> Self {
        fold_max(scope, self, rhs)
    }
    fn __expand_cast_method<To: Int>(self, scope: &Scope) -> NativeExpand<To> {
        match constant(&self) {
            Some(v) => ExpandValue::constant(v.into(), To::elem_type(scope)).into(),
            None => To::__expand_cast_from(scope, self),
        }
    }
    fn __expand_constant_method(self, _scope: &Scope) -> Option<u64> {
        constant(&self)
    }
}

/// Expand twin of [`IntegerSeq`]; blanket on integer sequences.
pub(crate) trait IntegerSeqExpand<C: Int>: Sized {
    fn __expand_sum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C>;
}

impl<C: Int> IntegerSeqExpand<C> for SequenceExpand<C> {
    fn __expand_sum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            ExpandValue::constant(0u64.into(), C::elem_type(scope)).into();
        for i in picks {
            let e = *self.__expand_index_method(scope, NativeExpand::from_lit(scope, i));
            acc = fold_add(scope, acc, e);
        }
        acc
    }
}
