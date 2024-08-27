use cubecl_macros_2::{expand_impl, Expand};
use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{
    new_ir::{Expr, IndexExpr, Integer, SliceExpr, SliceRangeExpr, SquareType, Strided},
    unexpanded,
};

use super::{Container, Dim1, Slice};

#[derive(new, Expand)]
#[expand(ir_type = T::ir_type())]
pub struct Array<T: SquareType> {
    _ty: PhantomData<T>,
}

impl<T: SquareType> Strided for Array<T> {
    type Dims = Dim1;
}

impl<T: SquareType> Container for Array<T> {
    type Item = T;
}

impl<T: SquareType, Idx: Integer> Index<Idx> for Array<T> {
    type Output = T;

    fn index(&self, _index: Idx) -> &Self::Output {
        unexpanded!()
    }
}

#[expand_impl]
impl<T: SquareType> Array<T> {
    #[expanded]
    pub fn index<Idx: Expr>(self, index: Idx) -> impl Expr<Output = T>
    where
        Idx::Output: Integer,
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

impl<T: SquareType, Idx: Integer> IndexMut<Idx> for Array<T> {
    fn index_mut(&mut self, _index: Idx) -> &mut Self::Output {
        unexpanded!()
    }
}
