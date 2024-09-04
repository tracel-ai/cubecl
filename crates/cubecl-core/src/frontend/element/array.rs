use std::{marker::PhantomData, num::NonZero};

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::Item,
    new_ir::{
        Container, Expand, Expanded, Expression, StaticExpand, StaticExpanded, Vectorization,
    },
    prelude::*,
    unexpanded, KernelSettings, Runtime,
};

use super::{
    ArgSettings, Dim1, Integer, LaunchArg, LaunchArgExpand, Primitive, Slice, TensorHandleRef,
};

use crate::new_ir::{
    EqExpr, Expr, GlobalVariable, IndexExpr, Length, SliceExpr, SliceRangeExpr, SquareType, Strided,
};
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

pub struct Array<T: SquareType> {
    size: u32,
    vectorization: Vectorization,
    _type: PhantomData<T>,
}
pub struct ArrayExpand<T: SquareType, Inner: Expr<Output = Array<T>>>(Inner);

unsafe impl<T: SquareType> Send for Array<T> {}
unsafe impl<T: SquareType> Sync for Array<T> {}

impl<T: SquareType> Expand for Array<T> {
    type Expanded<Inner: Expr<Output = Self>> = ArrayExpand<T, Inner>;

    fn expand<Inner: Expr<Output = Self>>(inner: Inner) -> Self::Expanded<Inner> {
        ArrayExpand(inner)
    }
}
impl<T: SquareType, Inner: Expr<Output = Array<T>>> Expanded for ArrayExpand<T, Inner> {
    type Unexpanded = Array<T>;

    fn inner(self) -> impl Expr<Output = Self::Unexpanded> {
        self.0
    }
}

impl<T: SquareType> StaticExpand for Array<T> {
    type Expanded = Self;
}
impl<T: SquareType> StaticExpanded for Array<T> {
    type Unexpanded = Self;
}

impl<T: SquareType> SquareType for Array<T> {
    fn ir_type() -> crate::ir::Elem {
        T::ir_type()
    }
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
impl<T: Primitive> Array<T> {
    pub fn new(size: u32) -> Self {
        Array {
            size,
            vectorization: None,
            _type: PhantomData,
        }
    }

    pub fn vectorized(size: u32, vectorization: u8) -> Self {
        Array {
            size,
            vectorization: NonZero::new(vectorization),
            _type: PhantomData,
        }
    }

    pub fn len(&self) -> u32 {
        self.size
    }

    #[expanded]
    pub fn len(self) -> impl Expr<Output = u32> {
        Length::new(self.0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[expanded]
    pub fn is_empty(self) -> impl Expr<Output = bool> {
        EqExpr::new(self.len(), 0)
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

impl<T: SquareType> Expr for Array<T> {
    type Output = Array<T>;

    fn expression_untyped(&self) -> Expression {
        Expression::ArrayInit {
            size: self.size,
            ty: T::ir_type(),
            vectorization: self.vectorization,
        }
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

impl<T: SquareType> Expr for &Array<T> {
    type Output = Array<T>;

    fn expression_untyped(&self) -> Expression {
        Array::<T>::expression_untyped(self)
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

impl<T: SquareType> Expr for &mut Array<T> {
    type Output = Array<T>;

    fn expression_untyped(&self) -> Expression {
        Array::<T>::expression_untyped(self)
    }

    fn vectorization(&self) -> Option<std::num::NonZero<u8>> {
        self.vectorization
    }
}

impl<T: SquareType, Idx: Integer> IndexMut<Idx> for Array<T> {
    fn index_mut(&mut self, _index: Idx) -> &mut Self::Output {
        unexpanded!()
    }
}

macro_rules! slice_impl {
    ($range:ident) => {
        impl<T: SquareType, Idx: Integer> Index<$range<Idx>> for Array<T> {
            type Output = Slice<Self, Idx>;

            fn index(&self, _index: $range<Idx>) -> &Self::Output {
                unexpanded!()
            }
        }

        impl<T: SquareType, Idx: Integer> IndexMut<$range<Idx>> for Array<T> {
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

impl<T: SquareType> Index<RangeFull> for Array<T> {
    type Output = Slice<Self, u32>;

    fn index(&self, _index: RangeFull) -> &Self::Output {
        unexpanded!()
    }
}
impl<T: SquareType> IndexMut<RangeFull> for Array<T> {
    fn index_mut(&mut self, _index: RangeFull) -> &mut Self::Output {
        unexpanded!()
    }
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle).
pub struct ArrayHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle<R::Server>,
    pub(crate) length: [usize; 1],
}

pub enum ArrayArg<'a, R: Runtime> {
    /// The array is passed with an array handle.
    Handle {
        /// The array handle.
        handle: ArrayHandleRef<'a, R>,
        /// The vectorization factor.
        vectorization_factor: u8,
    },
    /// The array is aliasing another input array.
    Alias {
        /// The position of the input array.
        input_pos: usize,
    },
}

impl<'a, R: Runtime> ArgSettings<R> for ArrayArg<'a, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        if let ArrayArg::Handle {
            handle,
            vectorization_factor: _,
        } = self
        {
            launcher.register_array(handle)
        }
    }

    fn configure_input(&self, position: usize, settings: KernelSettings) -> KernelSettings {
        match self {
            Self::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_input(position, *vectorization_factor),
            Self::Alias { input_pos: _ } => {
                panic!("Not yet supported, only output can be aliased for now.");
            }
        }
    }

    fn configure_output(&self, position: usize, mut settings: KernelSettings) -> KernelSettings {
        match self {
            Self::Handle {
                handle: _,
                vectorization_factor,
            } => settings.vectorize_output(position, *vectorization_factor),
            Self::Alias { input_pos } => {
                settings.mappings.push(crate::InplaceMapping {
                    pos_input: *input_pos,
                    pos_output: position,
                });
                settings
            }
        }
    }
}

impl<'a, R: Runtime> ArrayArg<'a, R> {
    /// Create a new array argument.
    ///
    /// # Safety
    ///
    /// Specifying the wrong lenght may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle<R::Server>,
        length: usize,
        vectorization_factor: u8,
    ) -> Self {
        ArrayArg::Handle {
            handle: ArrayHandleRef::from_raw_parts(handle, length),
            vectorization_factor,
        }
    }
}

impl<'a, R: Runtime> ArrayHandleRef<'a, R> {
    /// Create a new array handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong lenght may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle<R::Server>,
        length: usize,
    ) -> Self {
        Self {
            handle,
            length: [length],
        }
    }

    /// Return the handle as a tensor instead of an array.
    pub fn as_tensor(&self) -> TensorHandleRef<'_, R> {
        let shape = &self.length;

        TensorHandleRef {
            handle: self.handle,
            strides: &[1],
            shape,
        }
    }
}
