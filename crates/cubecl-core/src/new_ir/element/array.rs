use cubecl_macros_2::{expand_impl, Expand};
use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{
    ir::Item,
    new_ir::{
        Expr, GlobalVariable, IndexExpr, Integer, KernelBuilder, LaunchArg, LaunchArgExpand,
        Primitive, SliceExpr, SliceRangeExpr, SquareType, Strided,
    },
    prelude::ArrayArg,
    unexpanded, Runtime,
};

use super::{Container, Dim1, Slice};

#[derive(new, Expand)]
#[expand(ir_type = T::ir_type())]
pub struct Array<T: SquareType> {
    _ty: PhantomData<T>,
}

unsafe impl<T: SquareType> Send for Array<T> {}
unsafe impl<T: SquareType> Sync for Array<T> {}

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

impl<T: Primitive> LaunchArg for Array<T> {
    type RuntimeArg<'a, R: Runtime> = ArrayArg<'a, R>;
}

impl<T: Primitive> LaunchArgExpand for Array<T> {
    fn expand(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self> {
        builder.input_array(Item::vectorized(T::ir_type(), vectorization))
    }
    fn expand_output(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self> {
        builder.output_array(Item::vectorized(T::ir_type(), vectorization))
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
