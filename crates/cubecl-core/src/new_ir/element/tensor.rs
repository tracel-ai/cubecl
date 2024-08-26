use crate::new_ir::{
    Expand, Expr, IndexExpr, Integer, Length, Rank, Shape, SliceExpr, SliceRangeExpr, Stride,
    Strided,
};
use crate::{frontend::UInt, ir::Elem, new_ir::SquareType, unexpanded, Runtime};
use std::{
    marker::PhantomData,
    ops::{
        Index, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
};

pub struct Dyn;
pub struct Dim1;
pub struct Dim2;
pub struct Dim3;
pub struct Dim4;
pub struct Dim5;
pub struct Dim6;

pub type Tensor1<T> = Tensor<T, Dim1>;
pub type Tensor2<T> = Tensor<T, Dim2>;
pub type Tensor3<T> = Tensor<T, Dim3>;
pub type Tensor4<T> = Tensor<T, Dim4>;
pub type Tensor5<T> = Tensor<T, Dim5>;
pub type Tensor6<T> = Tensor<T, Dim6>;

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: SquareType, Dimensionality = Dyn> {
    _val: PhantomData<T>,
    _dim: PhantomData<Dimensionality>,
}

impl<T: SquareType, Dim> SquareType for Tensor<T, Dim> {
    fn ir_type() -> Elem {
        <T as SquareType>::ir_type()
    }
}

impl<T: SquareType, Dims> Expr for &Tensor<T, Dims> {
    type Output = Tensor<T, Dims>;

    fn expression_untyped(&self) -> crate::new_ir::Expression {
        panic!("Can't expand struct directly");
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

impl<T: SquareType, Dims> Expr for &mut Tensor<T, Dims> {
    type Output = Tensor<T, Dims>;

    fn expression_untyped(&self) -> crate::new_ir::Expression {
        panic!("Can't expand struct directly");
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        None
    }
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle),1``
/// the strides and the shape.
pub struct TensorHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle<R::Server>,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
}

impl<'a, R: Runtime> TensorHandleRef<'a, R> {
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn as_tensor_arg(&'a self, vectorisation: u8) -> TensorArg<'a, R> {
        unsafe { TensorArg::from_raw_parts(self.handle, self.strides, self.shape, vectorisation) }
    }
    /// Create a handle from raw parts.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle<R::Server>,
        strides: &'a [usize],
        shape: &'a [usize],
    ) -> Self {
        Self {
            handle,
            strides,
            shape,
        }
    }
}

/// Argument to be used for [tensors](Tensor) passed as arguments to kernels.
pub enum TensorArg<'a, R: Runtime> {
    /// The tensor is passed with a tensor handle.
    Handle {
        /// The tensor handle.
        handle: TensorHandleRef<'a, R>,
        /// The vectorization factor.
        vectorization_factor: u8,
    },
    /// The tensor is aliasing another input tensor.
    Alias {
        /// The position of the input tensor.
        input_pos: usize,
    },
}

impl<'a, R: Runtime> TensorArg<'a, R> {
    /// Create a new tensor argument specified with its vectorization factor.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle<R::Server>,
        strides: &'a [usize],
        shape: &'a [usize],
        factor: u8,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(handle, strides, shape),
                vectorization_factor: factor,
            }
        }
    }

    /// Create an alias argument.
    pub fn alias(position: usize) -> Self {
        Self::Alias {
            input_pos: position,
        }
    }
}

impl<T: SquareType, Dim> Tensor<T, Dim> {
    /// Obtain the stride of input at dimension dim
    pub fn stride<C: Integer>(&self, _dim: C) -> UInt {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape<C: Integer>(&self, _dim: C) -> UInt {
        unexpanded!()
    }

    /// The length of the buffer representing the tensor.
    ///
    /// # Warning
    ///
    /// The length will be affected by the vectorization factor. To obtain the number of elements,
    /// you should multiply the length by the vectorization factor.
    pub fn len(&self) -> UInt {
        unexpanded!()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> UInt {
        unexpanded!()
    }
}

pub struct TensorExpand<T: SquareType, Dim, Inner: Expr<Output = Tensor<T, Dim>>>(Inner);

impl<T: SquareType, Dim> Expand for Tensor<T, Dim> {
    type Expanded<Inner: Expr<Output = Self>> = TensorExpand<T, Dim, Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        TensorExpand(inner)
    }
}

impl<T: SquareType, Dimensions> Strided for Tensor<T, Dimensions> {}

impl<T: SquareType, Dimensions, Inner: Expr<Output = Tensor<T, Dimensions>>>
    TensorExpand<T, Dimensions, Inner>
{
    // Expanded version of stride
    pub fn stride<Dim: Expr>(self, dim: Dim) -> impl Expr<Output = Dim::Output>
    where
        Dim::Output: Integer,
    {
        Stride::new(self.0, dim)
    }

    // Expanded version of shape
    pub fn shape<Dim: Expr>(self, dim: Dim) -> impl Expr<Output = Dim::Output>
    where
        Dim::Output: Integer,
    {
        Shape::new(self.0, dim)
    }

    // Expanded version of len
    pub fn len<Out: Integer>(self) -> impl Expr<Output = Out> {
        Length::new(self.0)
    }

    // Expanded version of rank.
    pub fn rank<Out: Integer>(self) -> impl Expr<Output = Out> {
        Rank::new(self.0)
    }
}

impl<T: SquareType, Dims, Idx: Integer> Index<Idx> for Tensor<T, Dims> {
    type Output = T;

    fn index(&self, _index: Idx) -> &Self::Output {
        unexpanded!()
    }
}

impl<T: SquareType, Dims, Idx: Integer> IndexMut<Idx> for Tensor<T, Dims> {
    fn index_mut(&mut self, _index: Idx) -> &mut Self::Output {
        unexpanded!()
    }
}

impl<T: SquareType, Dims, Inner: Expr<Output = Tensor<T, Dims>>> TensorExpand<T, Dims, Inner> {
    pub fn index<Idx: Expr>(self, index: Idx) -> impl Expr<Output = T>
    where
        Inner::Output: Index<Idx::Output>,
        Idx::Output: Integer,
    {
        IndexExpr::new(self.0, index)
    }

    pub fn slice<TNum: Integer>(
        self,
        ranges: Vec<Box<dyn Expr<Output = SliceRangeExpr<TNum>>>>,
    ) -> impl Expr<Output = Inner::Output> {
        SliceExpr::new(self.0, ranges)
    }
}

macro_rules! slice_impl {
    ($range:ident) => {
        impl<T: SquareType, Dims, Idx: Integer> Index<$range<Idx>> for Tensor<T, Dims> {
            type Output = Self;

            fn index(&self, _index: $range<Idx>) -> &Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $range:ident, $dim_count:literal) => {
        impl<T: SquareType, Idx: Integer> Index<[$range<Idx>; $dim_count]> for Tensor<T, $dims> {
            type Output = Self;

            fn index(&self, _index: [$range<Idx>; $dim_count]) -> &Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $ty:ident, $($args:ident),*) => {
        impl<T: SquareType, $($args: RangeBounds<$ty>),*> Index<($($args),*)> for Tensor<T, $dims> {
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

        impl<T: SquareType, Dims> Index<RangeFull> for Tensor<T, Dims> {
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

        impl<T: SquareType> Index<[RangeFull; $dim_count]> for Tensor<T, $dims> {
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
        impl<T: SquareType, $dim, Idx: Integer> Index<[Idx; $num_dims]> for Tensor<T, $dim> {
            type Output = T;

            fn index(&self, _index: [Idx; $num_dims]) -> &Self::Output {
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
