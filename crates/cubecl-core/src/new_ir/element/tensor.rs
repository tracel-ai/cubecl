use crate::new_ir::{Expand, Expr, IndexExpr, Integer, Length, Rank, Shape, Stride, Strided};
use crate::{frontend::UInt, ir::Elem, new_ir::SquareType, unexpanded, Runtime};
use std::{marker::PhantomData, ops::Index};

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

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle),
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

pub struct TensorExpanded<T: SquareType, Dim, Inner: Expr<Output = Tensor<T, Dim>>>(Inner);

impl<T: SquareType, Dim> Expand for Tensor<T, Dim> {
    type Expanded<Inner: Expr<Output = Self>> = TensorExpanded<T, Dim, Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        TensorExpanded(inner)
    }
}

impl<T: SquareType, Dimensions> Strided for Tensor<T, Dimensions> {}

impl<T: SquareType, Dimensions, Inner: Expr<Output = Tensor<T, Dimensions>>>
    TensorExpanded<T, Dimensions, Inner>
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

impl<T: SquareType, Dims, Inner: Expr<Output = Tensor<T, Dims>>> TensorExpanded<T, Dims, Inner> {
    pub fn index<Idx: Expr>(self, index: Idx) -> impl Expr<Output = T>
    where
        Inner::Output: Index<Idx::Output>,
        Idx::Output: Integer,
    {
        IndexExpr::new(self.0, index)
    }
}

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
