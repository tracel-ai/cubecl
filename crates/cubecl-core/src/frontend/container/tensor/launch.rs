use std::{marker::PhantomData, num::NonZero};

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::{Item, Vectorization},
    prelude::{ArgSettings, CubePrimitive, ExpandElementTyped, LaunchArg, LaunchArgExpand},
    Runtime,
};

use super::Tensor;

/// Argument to be used for [tensors](Tensor) passed as arguments to kernels.
#[derive(Debug)]
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

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle),
/// the strides and the shape.
pub struct TensorHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
    pub runtime: PhantomData<R>,
}

impl<'a, R: Runtime> core::fmt::Debug for TensorHandleRef<'a, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "TensorHandleRef {{ strides: {:?}, shape: {:?} }}",
            self.strides, self.shape
        )
    }
}

/// Compilation argument for a [tensor](Tensor).
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
            Some(id) => builder.inplace_output(id).into(),
            None => builder
                .output_array(Item::vectorized(C::as_elem(), arg.vectorisation))
                .into(),
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
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

impl<'a, R: Runtime> TensorArg<'a, R> {
    /// Create a new tensor argument specified with its vectorization factor.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle,
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
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
    ) -> Self {
        Self {
            handle,
            strides,
            shape,
            runtime: PhantomData,
        }
    }
}
