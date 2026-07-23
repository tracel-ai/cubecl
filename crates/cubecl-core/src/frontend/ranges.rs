use alloc::boxed::Box;
use core::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use pliron::{
    builtin::op_interfaces::OneRegionInterface,
    irbuild::{
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    op::Op,
    region::Region,
};

use cubecl_ir::{
    ExpandState, ExpandValue, OpInserter,
    dialect::{
        branch::{RangeLoopOp, WhileOp},
        general::BoolAndOp,
    },
    types::scalar::BoolType,
};
use num_traits::NumCast;

use crate as cubecl;
use cubecl::prelude::*;

#[derive_expand(CubeType)]
pub struct Range<Idx: Scalar> {
    start: Idx,
    end: Idx,
}

impl<Idx: Scalar> RangeExpand<Idx> {
    pub fn new(start: NativeExpand<Idx>, end: NativeExpand<Idx>) -> Self {
        Self { start, end }
    }
}

impl<I: Int> RangeExpand<I> {
    pub fn __expand_step_by_method(
        self,
        _: &Scope,
        n: impl Into<NativeExpand<I>>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start: self.start,
            end: self.end,
            step: n.into(),
            inclusive: false,
        }
    }
}

impl<I: Int> Iterable for RangeExpand<I> {
    type Item = NativeExpand<I>;

    fn expand_unroll(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        iter_expand_unroll(scope, self.start, self.end, false, body);
    }

    fn expand(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        iter_expand(scope, self.start, self.end, false, body);
    }

    fn const_len(&self) -> Option<usize> {
        let start = self.start.expand.as_const()?.as_i64();
        let end = self.end.expand.as_const()?.as_i64();
        Some(start.abs_diff(end) as usize)
    }
}

impl IntoSliceIndices for RangeExpand<usize> {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        _scope: &Scope,
        _list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>) {
        (self.start, self.end)
    }
}

#[derive_expand(CubeType)]
pub struct RangeFrom<Idx: Scalar> {
    start: Idx,
}

impl<Idx: Scalar> RangeFromExpand<Idx> {
    pub fn new(start: NativeExpand<Idx>) -> Self {
        Self { start }
    }
}

impl IntoSliceIndices for RangeFromExpand<usize> {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        scope: &Scope,
        list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>) {
        let end = list.__expand_len_method(scope);
        (self.start, end)
    }
}

#[derive_expand(CubeType)]
pub struct RangeFull;

impl IntoSliceIndices for RangeFullExpand {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        scope: &Scope,
        list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>) {
        let start = NativeExpand::from_lit(scope, 0);
        let end = list.__expand_len_method(scope);
        (start, end)
    }
}

#[derive_expand(CubeType)]
pub struct RangeInclusive<Idx: Scalar> {
    start: Idx,
    last: Idx,
}

impl<Idx: Scalar> RangeInclusiveExpand<Idx> {
    pub fn new(start: NativeExpand<Idx>, last: NativeExpand<Idx>) -> Self {
        Self { start, last }
    }
}

impl<I: Int> RangeInclusiveExpand<I> {
    pub fn __expand_step_by_method(
        self,
        _: &Scope,
        n: impl Into<NativeExpand<I>>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start: self.start,
            end: self.last,
            step: n.into(),
            inclusive: true,
        }
    }
}

impl<I: Int> Iterable for RangeInclusiveExpand<I> {
    type Item = NativeExpand<I>;

    fn expand_unroll(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        iter_expand_unroll(scope, self.start, self.last, true, body);
    }

    fn expand(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        iter_expand(scope, self.start, self.last, true, body);
    }

    fn const_len(&self) -> Option<usize> {
        let start = self.start.expand.as_const()?.as_i64();
        let end = self.last.expand.as_const()?.as_i64();
        Some(start.abs_diff(end) as usize + 1)
    }
}

impl IntoSliceIndices for RangeInclusiveExpand<usize> {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        scope: &Scope,
        _list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>) {
        let end = self
            .last
            .__expand_add_method(scope, NativeExpand::from_lit(scope, 1));
        (self.start, end)
    }
}

#[derive_expand(CubeType)]
pub struct RangeTo<Idx: Scalar> {
    end: Idx,
}

impl<Idx: Scalar> RangeToExpand<Idx> {
    pub fn new(end: NativeExpand<Idx>) -> Self {
        Self { end }
    }
}

impl<I: Int> RangeToExpand<I> {
    pub fn __expand_step_by_method(
        self,
        scope: &Scope,
        n: impl Into<NativeExpand<I>>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start: NativeExpand::from_lit(scope, I::new(0)),
            end: self.end,
            step: n.into(),
            inclusive: false,
        }
    }
}

impl<I: Int> Iterable for RangeToExpand<I> {
    type Item = NativeExpand<I>;

    fn expand_unroll(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        let start = NativeExpand::from_lit(scope, I::new(0));
        iter_expand_unroll(scope, start, self.end, false, body);
    }

    fn expand(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        let start = NativeExpand::from_lit(scope, I::new(0));
        iter_expand(scope, start, self.end, false, body);
    }

    fn const_len(&self) -> Option<usize> {
        Some(self.end.expand.as_const()?.as_usize())
    }
}

impl IntoSliceIndices for RangeToExpand<usize> {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        scope: &Scope,
        _list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>) {
        let start = NativeExpand::from_lit(scope, 0);
        (start, self.end)
    }
}

#[derive_expand(CubeType)]
pub struct RangeToInclusive<Idx: Scalar> {
    last: Idx,
}

impl<Idx: Scalar> RangeToInclusiveExpand<Idx> {
    pub fn new(last: NativeExpand<Idx>) -> Self {
        Self { last }
    }
}

impl<I: Int> RangeToInclusiveExpand<I> {
    pub fn __expand_step_by_method(
        self,
        scope: &Scope,
        n: impl Into<NativeExpand<I>>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start: NativeExpand::from_lit(scope, I::new(0)),
            end: self.last,
            step: n.into(),
            inclusive: true,
        }
    }
}

impl<I: Int> Iterable for RangeToInclusiveExpand<I> {
    type Item = NativeExpand<I>;

    fn expand_unroll(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        let start = NativeExpand::from_lit(scope, I::new(0));
        iter_expand_unroll(scope, start, self.last, true, body);
    }

    fn expand(self, scope: &Scope, body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        let start = NativeExpand::from_lit(scope, I::new(0));
        iter_expand(scope, start, self.last, true, body);
    }

    fn const_len(&self) -> Option<usize> {
        Some(self.last.expand.as_const()?.as_usize() + 1)
    }
}

impl IntoSliceIndices for RangeToInclusiveExpand<usize> {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        scope: &Scope,
        _list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>) {
        let start = NativeExpand::from_lit(scope, 0);
        let end = self
            .last
            .__expand_add_method(scope, NativeExpand::from_lit(scope, 1));
        (start, end)
    }
}

pub(crate) trait IntoSliceIndices {
    fn into_slice_indices<E: CubePrimitive>(
        self,
        scope: &Scope,
        list: &impl ListExpand<E>,
    ) -> (NativeExpand<usize>, NativeExpand<usize>);
}

macro_rules! impl_slice_ranges {
    ($ty: ty, $range: ty) => {
        impl<E: CubePrimitive> IndexExpand<$range> for $ty {
            type Output = SliceExpand<E>;

            fn __expand_index_method(&self, scope: &Scope, index: $range) -> &Self::Output {
                let (start, end) = index.into_slice_indices(scope, self);
                self.__expand_slice_method(scope, start, end)
            }
        }

        impl<E: CubePrimitive> IndexMutExpand<$range> for $ty {
            fn __expand_index_mut_method(
                &mut self,
                scope: &Scope,
                index: $range,
            ) -> &mut <Self as IndexExpand<$range>>::Output {
                let (start, end) = index.into_slice_indices(scope, self);
                self.__expand_slice_mut_method(scope, start, end)
            }
        }
    };
    ($ty: ty) => {
        impl_slice_ranges!($ty, RangeExpand<usize>);
        impl_slice_ranges!($ty, RangeFromExpand<usize>);
        impl_slice_ranges!($ty, RangeFullExpand);
        impl_slice_ranges!($ty, RangeInclusiveExpand<usize>);
        impl_slice_ranges!($ty, RangeToExpand<usize>);
        impl_slice_ranges!($ty, RangeToInclusiveExpand<usize>);
    };
}
pub(crate) use impl_slice_ranges;

fn iter_expand_unroll<I: Int>(
    scope: &Scope,
    start: NativeExpand<I>,
    end: NativeExpand<I>,
    inclusive: bool,
    mut body: impl FnMut(&Scope, <I as CubeType>::ExpandType),
) {
    let start = start
        .expand
        .as_const()
        .expect("Only constant start can be unrolled.")
        .as_i64();
    let end = end
        .expand
        .as_const()
        .expect("Only constant end can be unrolled.")
        .as_i64();

    if inclusive {
        for i in start..=end {
            let val = I::from_int(i);
            body(scope, val.into())
        }
    } else {
        for i in start..end {
            let val = I::from_int(i);
            body(scope, val.into())
        }
    }
}

fn iter_expand<I: Int>(
    scope: &Scope,
    start: NativeExpand<I>,
    end: NativeExpand<I>,
    inclusive: bool,
    mut body: impl FnMut(&Scope, <I as CubeType>::ExpandType),
) {
    let start = I::__expand_cast_from(scope, start).expand;
    let mut end = I::__expand_cast_from(scope, end);
    let step: ExpandValue = I::new(1).into();

    if inclusive {
        end = end.__expand_add_method(scope, I::new(1).into());
    }

    let index_ty = I::__expand_as_type(scope);
    let start = start.read_value(scope);
    let end = end.read_value(scope);
    let step = step.read_value(scope);

    let i = scope.create_local_mut(index_ty, None);
    let range_loop = RangeLoopOp::new(scope.ctx_mut(), i, start, end, step);
    let body_block = range_loop.loop_body(scope.ctx());
    let child = scope.loop_child(OpInserter::new_at_block_end(body_block));

    body(&child, i.into());
    child.terminate_yield();

    register_range_loop::<I>(scope, &range_loop, &child);
    scope.set_may_return(&[child]);
}

pub struct SteppedRangeExpand<I: Int> {
    start: NativeExpand<I>,
    end: NativeExpand<I>,
    step: NativeExpand<I>,
    inclusive: bool,
}

impl<I: Int + Into<ExpandValue>> Iterable for SteppedRangeExpand<I> {
    type Item = NativeExpand<I>;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, <I as CubeType>::ExpandType)) {
        let index_ty = I::__expand_as_type(scope);

        let mut end = self.end;
        if self.inclusive {
            end = end.__expand_add_method(scope, I::new(1).into());
        }

        let start = self.start.read_value(scope);
        let end = end.read_value(scope);
        let step = self.step.read_value(scope);

        let i = scope.create_local_mut(index_ty, None);
        let range_loop = RangeLoopOp::new(scope.ctx_mut(), i, start, end, step);
        let body_block = range_loop.loop_body(scope.ctx());
        let child = scope.loop_child(OpInserter::new_at_block_end(body_block));

        body(&child, i.into());

        register_range_loop::<I>(scope, &range_loop, &child);
        scope.set_may_return(&[child]);
    }

    fn expand_unroll(
        self,
        scope: &Scope,
        mut body: impl FnMut(&Scope, <I as CubeType>::ExpandType),
    ) {
        let start = self
            .start
            .expand
            .as_const()
            .expect("Only constant start can be unrolled.")
            .as_i128();
        let end = self
            .end
            .expand
            .as_const()
            .expect("Only constant end can be unrolled.")
            .as_i128();
        let step = self
            .step
            .expand
            .as_const()
            .expect("Only constant step can be unrolled.")
            .as_i128();

        match (self.inclusive, step.is_negative()) {
            (true, true) => {
                for i in (end..=start).rev().step_by(step.unsigned_abs() as usize) {
                    let val = I::from_int_128(i);
                    body(scope, val.into())
                }
            }
            (true, false) => {
                for i in (start..=end).step_by(step.unsigned_abs() as usize) {
                    let val = I::from_int_128(i);
                    body(scope, val.into())
                }
            }
            (false, true) => {
                for i in (end..start).rev().step_by(step.unsigned_abs() as usize) {
                    let val = I::from_int_128(i);
                    body(scope, val.into())
                }
            }
            (false, false) => {
                for i in (start..end).step_by(step.unsigned_abs() as usize) {
                    let val = I::from_int_128(i);
                    body(scope, val.into())
                }
            }
        }
    }

    fn const_len(&self) -> Option<usize> {
        let start = self.start.constant()?.as_i128();
        let end = self.end.constant()?.as_i128();
        let step = self.step.constant()?.as_i128().unsigned_abs();
        Some((start.abs_diff(end) / step) as usize)
    }
}

/// integer range. Equivalent to:
///
/// ```ignore
/// start..end
/// ```
pub fn range<T: Int>(start: T, end: T) -> impl Iterator<Item = T> {
    let start: i64 = start.to_i64().unwrap();
    let end: i64 = end.to_i64().unwrap();
    (start..end).map(<T as NumCast>::from).map(Option::unwrap)
}

pub mod range {
    use cubecl_ir::Scope;

    use crate::prelude::{Int, NativeExpand};

    use super::RangeExpand;

    pub fn expand<I: Int>(
        _scope: &Scope,
        start: NativeExpand<I>,
        end: NativeExpand<I>,
    ) -> RangeExpand<I> {
        RangeExpand { start, end }
    }
}

/// Stepped range. Equivalent to:
///
/// ```ignore
/// (start..end).step_by(step)
/// ```
///
/// Allows using any integer for the step, instead of just usize
pub fn range_stepped<I: Int>(start: I, end: I, step: I) -> Box<dyn Iterator<Item = I>> {
    let start = start.to_i128().unwrap();
    let end = end.to_i128().unwrap();
    let step = step.to_i128().unwrap();

    if step < 0 {
        Box::new(
            (end..start)
                .rev()
                .step_by(step.unsigned_abs() as usize)
                .map(<I as NumCast>::from)
                .map(Option::unwrap),
        )
    } else {
        Box::new(
            (start..end)
                .step_by(step.unsigned_abs() as usize)
                .map(<I as NumCast>::from)
                .map(Option::unwrap),
        )
    }
}

pub mod range_stepped {
    use cubecl_ir::Scope;

    use crate::prelude::{Int, NativeExpand};

    use super::SteppedRangeExpand;

    pub fn expand<I: Int>(
        _scope: &Scope,
        start: NativeExpand<I>,
        end: NativeExpand<I>,
        step: NativeExpand<I>,
    ) -> SteppedRangeExpand<I> {
        SteppedRangeExpand {
            start,
            end,
            step,
            inclusive: false,
        }
    }
}

/// register a range loop if it contains no break or return, destructure to while if it does
pub(crate) fn register_range_loop<I: Int>(scope: &Scope, for_op: &RangeLoopOp, body: &Scope) {
    let ctx = scope.ctx_mut();
    let ExpandState {
        may_return,
        may_break,
        inv_return_flag,
        inv_break_flag,
    } = *body.expand_state();
    if !may_break && !may_return {
        body.terminate_yield();
        scope.register(for_op);
        return;
    }

    let start = for_op.start(ctx);
    let end = for_op.end(ctx);
    let step = for_op.step(ctx);
    let mut iter_var: NativeExpand<I> = for_op.iter_var(ctx).into();

    assign::expand_element(scope, start.into(), iter_var.expand);
    let initial_cond = I::__expand_native_lt(scope, iter_var.into(), end.into());
    let cond_ptr = scope.create_local_mut(BoolType::get(ctx), None);
    assign::expand_element(scope, initial_cond, cond_ptr.into());

    assign_binop_expand(body, &mut iter_var, step.into(), I::__expand_native_add);

    // Return/break cond and store need to go after the predication, comparison also goes there to
    // avoid an extra load/store
    body.inserter()
        .set_insertion_point_to_block_end(for_op.loop_body(ctx));
    let mut cond = I::__expand_native_lt(body, iter_var.into(), end.into());
    if may_break {
        cond = binary_expand(body, cond, inv_break_flag.unwrap().into(), BoolAndOp::new);
    }
    if may_return {
        let inv_return_flag = inv_return_flag.unwrap();
        cond = binary_expand(body, cond, inv_return_flag.into(), BoolAndOp::new);
    }
    assign::expand_element(body, cond, cond_ptr.into());
    body.terminate_yield();

    let mut rewriter = IRRewriter::<DummyListener>::default();
    let while_op = WhileOp::new(ctx, cond_ptr);

    rewriter.erase_region(ctx, while_op.get_region(ctx));
    Region::move_to_op(for_op.get_region(ctx), while_op.get_operation(), ctx);
    rewriter.erase_operation(ctx, for_op.get_operation());

    scope.register(&while_op);
}
