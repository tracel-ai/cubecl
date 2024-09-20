use super::{ExpandElementBaseInit, ExpandElementTyped, LaunchArgExpand, SizedContainer};
use crate::{
    frontend::{
        indexation::Index, ArgSettings, CubeContext, CubePrimitive, CubeType, ExpandElement,
    },
    ir::{Elem, Item, Metadata, Variable, Vectorization},
    prelude::{KernelBuilder, KernelLauncher},
    unexpanded, LaunchArg, Runtime,
};
use std::{marker::PhantomData, num::NonZero};

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<C: CubeType> ExpandElementBaseInit for Tensor<C> {
    fn init_elem(_context: &mut crate::prelude::CubeContext, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TensorCompilationArg {
    inplace: Option<u16>,
    vectorisation: Vectorization,
}

impl<C: CubePrimitive> LaunchArgExpand for Tensor<C> {
    type CompilationArg = TensorCompilationArg;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        builder
            .input_array(Item::vectorized(C::as_elem(), arg.vectorisation))
            .into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        match arg.inplace {
            Some(id) => builder
                .inplace_output(id, Item::vectorized(C::as_elem(), arg.vectorisation))
                .into(),
            None => builder
                .output_array(Item::vectorized(C::as_elem(), arg.vectorisation))
                .into(),
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;

    fn compilation_arg<'a, R: Runtime>(
        runtime_arg: &Self::RuntimeArg<'a, R>,
    ) -> Self::CompilationArg {
        match runtime_arg {
            TensorArg::Handle {
                handle: _,
                vectorization_factor,
            } => TensorCompilationArg {
                inplace: None,
                vectorisation: Vectorization::Some(NonZero::new(*vectorization_factor).unwrap()),
            },
            TensorArg::Alias { input_pos } => TensorCompilationArg {
                inplace: Some(*input_pos as u16),
                vectorisation: Vectorization::None,
            },
        }
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
}

impl<T: CubeType> Tensor<T> {
    /// Obtain the stride of input at dimension dim
    pub fn stride<C: Index>(&self, _dim: C) -> u32 {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape<C: Index>(&self, _dim: C) -> u32 {
        unexpanded!()
    }

    /// The length of the buffer representing the tensor.
    ///
    /// # Warning
    ///
    /// The length will be affected by the vectorization factor. To obtain the number of elements,
    /// you should multiply the length by the vectorization factor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unexpanded!()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unexpanded!()
    }
}

impl<T: CubeType> ExpandElementTyped<T> {
    // Expanded version of stride
    pub fn __expand_stride_method(
        self,
        context: &mut CubeContext,
        dim: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        let dim: ExpandElement = dim.into();
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Stride {
            dim: *dim,
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out.into()
    }

    // Expanded version of shape
    pub fn __expand_shape_method(
        self,
        context: &mut CubeContext,
        dim: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        let dim: ExpandElement = dim.into();
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Shape {
            dim: *dim,
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out.into()
    }

    // Expanded version of len
    pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
        let out = context.create_local(Item::new(Elem::UInt));
        context.register(Metadata::Length {
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out.into()
    }

    // Expanded version of rank.
    pub fn __expand_rank_method(self, _context: &mut CubeContext) -> ExpandElementTyped<u32> {
        ExpandElement::Plain(Variable::Rank).into()
    }
}

impl<T: CubeType<ExpandType = ExpandElementTyped<T>>> SizedContainer for Tensor<T> {
    type Item = T;
}

impl<T: CubeType> Iterator for &Tensor<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}
