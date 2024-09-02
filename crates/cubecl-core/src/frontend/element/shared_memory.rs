use std::{
    marker::PhantomData,
    num::NonZero,
    ops::{Index, IndexMut, Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive},
};

use crate::{
    frontend::CubeContext,
    ir::Elem,
    new_ir::{
        flatten::item, Container, Expand, Expanded, Expr, Expression, IndexExpr, SliceExpr,
        SliceRangeExpr, SquareType, StaticExpand, StaticExpanded, Strided, Vectorization,
    },
    prelude::*,
    unexpanded,
};

use super::{Dim1, ExpandElement, Integer, Primitive, Slice};

#[derive(Clone, Copy)]
pub struct SharedMemory<T: SquareType> {
    size: u32,
    vectorization: Vectorization,
    _type: PhantomData<T>,
}

#[derive(Clone, Copy)]
pub struct SharedMemoryExpand<T: SquareType, Inner: Expr<Output = SharedMemory<T>>>(Inner);

impl<T: SquareType> StaticExpand for SharedMemory<T> {
    type Expanded = Self;
}
impl<T: SquareType> StaticExpanded for SharedMemory<T> {
    type Unexpanded = Self;
}

impl<T: SquareType> Expand for SharedMemory<T> {
    type Expanded<Inner: Expr<Output = Self>> = SharedMemoryExpand<T, Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        SharedMemoryExpand(inner)
    }
}

impl<T: SquareType, Inner: Expr<Output = SharedMemory<T>>> Expanded
    for SharedMemoryExpand<T, Inner>
{
    type Unexpanded = SharedMemory<T>;

    fn inner(self) -> impl Expr<Output = Self::Unexpanded> {
        self.0
    }
}

impl<T: SquareType> SquareType for SharedMemory<T> {
    fn ir_type() -> Elem {
        T::ir_type()
    }
}

impl<T: SquareType> Strided for SharedMemory<T> {
    type Dims = Dim1;
}

impl<T: SquareType> Container for SharedMemory<T> {
    type Item = T;
}

#[derive(Clone, Debug, PartialEq)]
pub enum SharedMemoryExpr {
    Init {
        size: u32,
        ty: Elem,
        vectorization: Vectorization,
    },
}

impl SharedMemoryExpr {
    pub fn ir_type(&self) -> Elem {
        match self {
            SharedMemoryExpr::Init { ty, .. } => *ty,
        }
    }

    pub fn vectorization(&self) -> Vectorization {
        match self {
            SharedMemoryExpr::Init { vectorization, .. } => *vectorization,
        }
    }

    pub fn flatten(self, context: &mut CubeContext) -> Option<ExpandElement> {
        match self {
            SharedMemoryExpr::Init {
                size,
                ty,
                vectorization,
            } => {
                let var = context.create_shared(item(ty, vectorization), size);
                var.into()
            }
        }
    }
}

// #[derive(new)]
// pub struct SharedMemoryInit<T: SquareType> {
//     pub size: u32,
//     pub vectorization: Vectorization,
//     pub _type: PhantomData<T>,
// }

impl<T: SquareType> Expr for SharedMemory<T> {
    type Output = SharedMemory<T>;

    fn expression_untyped(&self) -> Expression {
        SharedMemoryExpr::Init {
            size: self.size,
            ty: T::ir_type(),
            vectorization: self.vectorization,
        }
        .into()
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

impl<T: SquareType> Expr for &SharedMemory<T> {
    type Output = SharedMemory<T>;

    fn expression_untyped(&self) -> Expression {
        SharedMemory::<T>::expression_untyped(self)
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

impl<T: SquareType> Expr for &mut SharedMemory<T> {
    type Output = SharedMemory<T>;

    fn expression_untyped(&self) -> Expression {
        SharedMemory::<T>::expression_untyped(self)
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

#[expand_impl]
impl<T: Primitive> SharedMemory<T> {
    pub fn new(size: u32) -> Self {
        SharedMemory {
            size,
            vectorization: None,
            _type: PhantomData,
        }
    }

    pub fn vectorized(size: u32, vectorization_factor: u32) -> Self {
        SharedMemory {
            size,
            vectorization: NonZero::new(vectorization_factor as u8),
            _type: PhantomData,
        }
    }

    #[expanded]
    pub fn index<Idx: Expr>(self, index: Idx) -> impl Expr<Output = T>
    where
        Idx::Output: Integer,
    {
        IndexExpr::new(self.0, index)
    }

    #[expanded]
    pub fn slice<Start: Expr>(
        self,
        ranges: Vec<Box<dyn Expr<Output = SliceRangeExpr<Start>>>>,
    ) -> impl Expr<Output = Slice<__Inner, Start::Output>>
    where
        Start::Output: Integer,
    {
        SliceExpr::new(self.0, ranges)
    }
}

macro_rules! slice_impl {
    ($range:ident) => {
        impl<T: SquareType, Idx: Integer> Index<$range<Idx>> for SharedMemory<T> {
            type Output = Slice<Self, Idx>;

            fn index(&self, _index: $range<Idx>) -> &Self::Output {
                unexpanded!()
            }
        }

        impl<T: SquareType, Idx: Integer> IndexMut<$range<Idx>> for SharedMemory<T> {
            fn index_mut(&mut self, _index: $range<Idx>) -> &mut Self::Output {
                unexpanded!()
            }
        }
    };
}

slice_impl!(Range);
slice_impl!(RangeFrom);
slice_impl!(RangeInclusive);
slice_impl!(RangeTo);
slice_impl!(RangeToInclusive);

impl<T: SquareType, I: Integer> Index<I> for SharedMemory<T> {
    type Output = T;

    fn index(&self, _index: I) -> &Self::Output {
        unexpanded!()
    }
}

impl<T: SquareType, I: Integer> IndexMut<I> for SharedMemory<T> {
    fn index_mut(&mut self, _index: I) -> &mut Self::Output {
        unexpanded!()
    }
}
