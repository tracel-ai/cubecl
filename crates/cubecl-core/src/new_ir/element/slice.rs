use std::{
    marker::PhantomData,
    ops::{
        Index, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
};

use cubecl_macros_2::{expand_impl, Expand};

use crate::{
    new_ir::{Expr, IndexExpr, Integer, SliceExpr, SliceRangeExpr, SquareType, Strided},
    unexpanded,
};

use super::{Container, Dim2, Dim3, Dim4, Dim5, Dim6};

#[derive(new, Expand)]
#[expand(ir_type = <Inner::Output as Container>::Item::ir_type())]
pub struct Slice<Inner: Expr, Num: Integer>
where
    Inner::Output: Strided + Container,
    <Inner::Output as Container>::Item: SquareType,
{
    #[expand(skip)]
    pub inner: Inner,
    pub _num: PhantomData<Num>,
}

impl<Inner: Expr, Num: Integer> Strided for Slice<Inner, Num>
where
    Inner::Output: Strided + Container,
    <Inner::Output as Container>::Item: SquareType,
{
    type Dims = <Inner::Output as Strided>::Dims;
}

impl<Inner: Expr, Num: Integer> Container for Slice<Inner, Num>
where
    Inner::Output: Strided + Container,
    <Inner::Output as Container>::Item: SquareType,
{
    type Item = <Inner::Output as Container>::Item;
}

#[expand_impl]
impl<Inner: Expr, Idx: Integer> Slice<Inner, Idx>
where
    Inner::Output: Strided + Container,
    <Inner::Output as Container>::Item: SquareType,
{
    #[expanded]
    pub fn index(
        self,
        index: impl Expr<Output = Idx>,
    ) -> impl Expr<Output = <Inner::Output as Container>::Item>
    where
        Inner::Output: Index<Idx>,
    {
        IndexExpr::new(self.0, index)
    }

    #[expanded]
    pub fn slice<TNum: Integer>(
        self,
        ranges: Vec<Box<dyn Expr<Output = SliceRangeExpr<TNum>>>>,
    ) -> impl Expr<Output = Slice<__Inner, TNum>> {
        SliceExpr::new(self.0, ranges)
    }
}

impl<Inner: Expr, Idx: Integer> Index<Idx> for Slice<Inner, Idx>
where
    Inner::Output: Strided + Container,
    <Inner::Output as Container>::Item: SquareType,
{
    type Output = <Inner::Output as Container>::Item;

    fn index(&self, _index: Idx) -> &Self::Output {
        unexpanded!()
    }
}

impl<Inner: Expr, Idx: Integer> IndexMut<Idx> for Slice<Inner, Idx>
where
    Inner::Output: Strided + Container,
    <Inner::Output as Container>::Item: SquareType,
{
    fn index_mut(&mut self, _index: Idx) -> &mut Self::Output {
        unexpanded!()
    }
}

macro_rules! slice_impl {
    ($range:ident) => {
        impl<Inner: Expr, Idx: Integer> Index<$range<Idx>> for Slice<Inner, Idx>
            where Inner::Output: Strided + Container,
            <Inner::Output as Container>::Item: SquareType
        {
            type Output = Self;

            fn index(&self, _index: $range<Idx>) -> &Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $range:ident, $dim_count:literal) => {
        impl<Inner: Expr, Idx: Integer> Index<[$range<Idx>; $dim_count]> for Slice<Inner, Idx>
            where Inner::Output: Strided<Dims = $dims> + Container,
            <Inner::Output as Container>::Item: SquareType
        {
            type Output = Self;

            fn index(&self, _index: [$range<Idx>; $dim_count]) -> &Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $ty:ident, $($args:ident),*) => {
        impl<Inner: Expr, $($args: RangeBounds<$ty>),*> Index<($($args),*)> for Slice<Inner, $ty>
            where Inner::Output: Strided<Dims = $dims> + Container,
            <Inner::Output as Container>::Item: SquareType
        {
            type Output = Self;

            fn index(&self, _index: ($($args),*)) -> &Self::Output {
                unexpanded!()
            }
        }
    };
}

macro_rules! slice_impls {
    () => {
        slice_impl!(Range);
        slice_impl!(RangeFrom);
        slice_impl!(RangeInclusive);
        slice_impl!(RangeTo);
        slice_impl!(RangeToInclusive);

        impl<Inner: Expr, Idx: Integer> Index<RangeFull> for Slice<Inner, Idx>
            where Inner::Output: Strided + Container,
            <Inner::Output as Container>::Item: SquareType
        {
            type Output = Self;

            fn index(&self, _index: RangeFull) -> &Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $dim_count:literal) => {
        slice_impl!($dims, Range, $dim_count);
        slice_impl!($dims, RangeFrom, $dim_count);
        slice_impl!($dims, RangeInclusive, $dim_count);
        slice_impl!($dims, RangeTo, $dim_count);
        slice_impl!($dims, RangeToInclusive, $dim_count);

        impl<Inner: Expr, Idx: Integer> Index<[RangeFull; $dim_count]> for Slice<Inner, Idx>
            where Inner::Output: Strided<Dims = $dims> + Container,
            <Inner::Output as Container>::Item: SquareType
        {
            type Output = Self;

            fn index(&self, _index: [RangeFull; $dim_count]) -> &Self::Output {
                unexpanded!()
            }
        }

    };
    ($dims:ident, $($args:ident),*) => {
        slice_impl!($dims, u32, $($args),*);
    };
}

slice_impls!();

macro_rules! impl_index_array {
    ($dim:ident, $num_dims:literal) => {
        impl<Inner: Expr, Idx: Integer> Index<[Idx; $num_dims]> for Slice<Inner, Idx>
        where
            Inner::Output: Strided<Dims = $dim> + Container,
            <Inner::Output as Container>::Item: SquareType,
        {
            type Output = <Inner::Output as Container>::Item;

            fn index(&self, _index: [Idx; $num_dims]) -> &Self::Output {
                unexpanded!()
            }
        }

        impl<Inner: Expr, Idx: Integer> IndexMut<[Idx; $num_dims]> for Slice<Inner, Idx>
        where
            Inner::Output: Strided<Dims = $dim> + Container,
            <Inner::Output as Container>::Item: SquareType,
        {
            fn index_mut(&mut self, _index: [Idx; $num_dims]) -> &mut Self::Output {
                unexpanded!()
            }
        }
    };
}

impl_index_array!(Dim2, 2);
impl_index_array!(Dim3, 3);
impl_index_array!(Dim4, 4);
impl_index_array!(Dim5, 5);
impl_index_array!(Dim6, 6);

slice_impls!(Dim2, 2);
slice_impls!(Dim3, 3);
slice_impls!(Dim4, 4);
slice_impls!(Dim5, 5);
slice_impls!(Dim6, 6);

slice_impls!(Dim2, Range1, Range2);
slice_impls!(Dim3, Range1, Range2, Range3);
slice_impls!(Dim4, Range1, Range2, Range3, Range4);
slice_impls!(Dim5, Range1, Range2, Range3, Range4, Range5);
slice_impls!(Dim6, Range1, Range2, Range3, Range4, Range5, Range6);
