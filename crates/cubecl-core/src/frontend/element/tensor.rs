use super::{Integer, LaunchArgExpand};
use crate::{
    frontend::ArgSettings, ir::Item, new_ir::*, prelude::*, unexpanded, KernelSettings, LaunchArg,
    Runtime,
};
use std::marker::PhantomData;
use std::ops::{
    Index, IndexMut, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
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
#[derive(new, Expand)]
#[expand(ir_type = T::ir_type())]
pub struct Tensor<T: SquareType, Dimensionality = Dyn> {
    _val: PhantomData<T>,
    _dim: PhantomData<Dimensionality>,
}

unsafe impl<T: SquareType, Dims> Send for Tensor<T, Dims> {}
unsafe impl<T: SquareType, Dims> Sync for Tensor<T, Dims> {}

impl<T: SquareType, Dims> Strided for Tensor<T, Dims> {
    type Dims = Dims;
}
impl<T: SquareType, Dims> Container for Tensor<T, Dims> {
    type Item = T;
}

impl<T: SquareType + 'static, Dims: 'static> LaunchArgExpand for Tensor<T, Dims> {
    fn expand(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self> {
        builder.input_array(Item::vectorized(T::ir_type(), vectorization))
    }
    fn expand_output(builder: &mut KernelBuilder, vectorization: u8) -> GlobalVariable<Self> {
        builder.output_array(Item::vectorized(T::ir_type(), vectorization))
    }
}

impl<T: SquareType + 'static, Dims: 'static> LaunchArg for Tensor<T, Dims> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;
}

#[expand_impl]
impl<T: SquareType, Dims> Tensor<T, Dims> {
    /// Obtain the stride of input at dimension dim
    pub fn stride<C: Integer>(&self, _dim: C) -> u32 {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape<C: Integer>(&self, _dim: C) -> u32 {
        unexpanded!()
    }

    /// The length of the buffer representing the tensor.
    ///
    /// # Warning
    ///
    /// The length will be affected by the vectorization factor. To obtain the number of elements,
    /// you should multiply the length by the vectorization factor.
    pub fn len(&self) -> u32 {
        unexpanded!()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unexpanded!()
    }

    // Expanded version of stride
    #[expanded]
    pub fn stride<Dim: Expr>(self, dim: Dim) -> impl Expr<Output = Dim::Output>
    where
        Dim::Output: Integer,
    {
        Stride::new(self.0, dim)
    }

    // Expanded version of shape
    #[expanded]
    pub fn shape<Dim: Expr>(self, dim: Dim) -> impl Expr<Output = Dim::Output>
    where
        Dim::Output: Integer,
    {
        Shape::new(self.0, dim)
    }

    // Expanded version of len
    #[expanded]
    pub fn len(self) -> impl Expr<Output = u32> {
        Length::new(self.0)
    }

    // Expanded version of len
    #[expanded]
    pub fn is_empty(self) -> impl Expr<Output = bool> {
        EqExpr::new(self.len(), 0)
    }

    // Expanded version of rank.
    #[expanded]
    pub fn rank(self) -> impl Expr<Output = u32> {
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

#[expand_impl]
impl<T: SquareType, Dims> Tensor<T, Dims> {
    #[expanded]
    pub fn index<Idx: Expr>(self, index: Idx) -> impl Expr<Output = T>
    where
        __Inner::Output: Index<Idx::Output>,
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

macro_rules! slice_impl {
    ($range:ident) => {
        impl<T: SquareType, Dims, Idx: Integer> Index<$range<Idx>> for Tensor<T, Dims> {
            type Output = Slice<Self, Idx>;

            fn index(&self, _index: $range<Idx>) -> &Self::Output {
                unexpanded!()
            }
        }

        impl<T: SquareType, Dims, Idx: Integer> IndexMut<$range<Idx>> for Tensor<T, Dims> {
            fn index_mut(&mut self, _index: $range<Idx>) -> &mut Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $range:ident, $dim_count:literal) => {
        impl<T: SquareType, Idx: Integer> Index<[$range<Idx>; $dim_count]> for Tensor<T, $dims> {
            type Output = Slice<Self, Idx>;

            fn index(&self, _index: [$range<Idx>; $dim_count]) -> &Self::Output {
                unexpanded!()
            }
        }

        impl<T: SquareType, Idx: Integer> IndexMut<[$range<Idx>; $dim_count]> for Tensor<T, $dims> {
            fn index_mut(&mut self, _index: [$range<Idx>; $dim_count]) -> &mut Self::Output {
                unexpanded!()
            }
        }
    };
    ($dims:ident, $ty:ident, $($args:ident),*) => {
        impl<T: SquareType, $($args: RangeBounds<$ty>),*> Index<($($args),*)> for Tensor<T, $dims> {
            type Output = Slice<Self, $ty>;

            fn index(&self, _index: ($($args),*)) -> &Self::Output {
                unexpanded!()
            }
        }
        impl<T: SquareType, $($args: RangeBounds<$ty>),*> IndexMut<($($args),*)> for Tensor<T, $dims> {
            fn index_mut(&mut self, _index: ($($args),*)) -> &mut Self::Output {
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
            type Output = Slice<Self, u32>;

            fn index(&self, _index: RangeFull) -> &Self::Output {
                unexpanded!()
            }
        }
        impl<T: SquareType, Dims> IndexMut<RangeFull> for Tensor<T, Dims> {
            fn index_mut(&mut self, _index: RangeFull) -> &mut Self::Output {
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
            type Output = Slice<Self, u32>;

            fn index(&self, _index: [RangeFull; $dim_count]) -> &Self::Output {
                unexpanded!()
            }
        }
        impl<T: SquareType> IndexMut<[RangeFull; $dim_count]> for Tensor<T, $dims> {
            fn index_mut(&mut self, _index: [RangeFull; $dim_count]) -> &mut Self::Output {
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

impl<'a, R: Runtime> ArgSettings<R> for TensorArg<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        if let TensorArg::Handle {
            handle,
            vectorization_factor: _,
        } = self
        {
            launcher.register_tensor(handle)
        }
    }

    fn configure_input(&self, position: usize, settings: KernelSettings) -> KernelSettings {
        match self {
            TensorArg::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_input(position, *vectorization_factor),
            TensorArg::Alias { input_pos: _ } => {
                panic!("Not yet supported, only output can be aliased for now.");
            }
        }
    }

    fn configure_output(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
        match self {
            TensorArg::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_output(position, *vectorization_factor),
            TensorArg::Alias { input_pos } => {
                settings.mappings.push(crate::InplaceMapping {
                    pos_input: *input_pos,
                    pos_output: position,
                });
                settings
            }
        }
    }
}
