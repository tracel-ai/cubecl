use core::marker::PhantomData;

use cubecl_ir::{AddressType, Id};
use cubecl_runtime::{runtime::Runtime, server::CopyDescriptor};
use cubecl_zspace::{Shape, Strides};

use crate::{
    self as cubecl,
    compute::{KernelBuilder, KernelLauncher},
    frontend::container::slice,
    prelude::*,
};

use super::Tensor;

#[derive(CubeType, CubeLaunch, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct TensorMeta {
    pub len: usize,
    pub rank: usize,
}

/// Argument to be used for [tensors](Tensor) passed as arguments to kernels.
#[derive(Debug)]
pub enum TensorArg<R: Runtime> {
    /// The tensor is passed with a tensor handle.
    Handle {
        /// The tensor handle.
        handle: TensorBinding<R>,
    },
    /// The tensor is aliasing another input tensor.
    Alias {
        /// The position of the input tensor.
        input_pos: usize,
        strides: Strides,
        shape: Shape,
    },
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle),
/// the strides and the shape.
pub struct TensorBinding<R: Runtime> {
    pub handle: cubecl_runtime::server::Binding,
    pub strides: Strides,
    pub shape: Shape,
    pub runtime: PhantomData<R>,
}

impl<R: Runtime> Clone for TensorBinding<R> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            strides: self.strides.clone(),
            shape: self.shape.clone(),
            runtime: PhantomData,
        }
    }
}

impl<R: Runtime> TensorBinding<R> {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Address type required to fully index this tensor handle, assuming scalar access.
    pub fn required_address_type(&self, elem_size: usize) -> AddressType {
        AddressType::from_len(self.handle.size() as usize / elem_size)
    }
}

impl<R: Runtime> core::fmt::Debug for TensorBinding<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
    pub meta: TensorMetaCompilationArg,
    pub buffer: BufferCompilationArg,
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<R: Runtime> = TensorArg<R>;
    type CompilationArg = TensorCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let ty = launcher.with_scope(|scope| C::__expand_as_type(scope));
        let len = arg.size() / ty.vector_size();
        let meta_arg = TensorMetaLaunch::new(len, arg.shape().len());
        let buffer = match &arg {
            TensorArg::Handle { .. } => BufferCompilationArg { inplace: None },
            TensorArg::Alias { input_pos, .. } => BufferCompilationArg {
                inplace: Some(*input_pos as Id),
            },
        };
        launcher.register_tensor(arg, ty);
        let meta = TensorMeta::register(meta_arg, launcher);
        TensorCompilationArg { meta, buffer }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TensorExpand<C> {
        let buffer = match arg.buffer.inplace {
            Some(id) => builder.inplace(id),
            None => builder.tensor(C::__expand_as_type(&builder.scope)),
        };
        let meta = TensorMeta::expand(&arg.meta, builder);
        let scope = &builder.scope;
        let len = expand_buffer_length_native(scope, buffer);
        let buffer =
            slice::from_raw_parts::<C>(scope, buffer, 0usize.into_expand(scope), len.into());
        TensorExpand { meta, buffer }
    }
}

impl<C: CubePrimitive> LaunchArg for OwnedTensor<C> {
    type RuntimeArg<R: Runtime> = TensorArg<R>;
    type CompilationArg = TensorCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        Tensor::<C>::register(arg, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> OwnedTensorExpand<C> {
        let tensor = Tensor::<C>::expand(arg, builder);
        OwnedTensorExpand {
            meta: tensor.meta,
            buffer: tensor.buffer.expand.into(),
        }
    }
}

impl<R: Runtime> TensorArg<R> {
    /// Create a new tensor argument specified with its vectorization factor.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts(
        handle: cubecl_runtime::server::Handle,
        strides: Strides,
        shape: Shape,
    ) -> Self {
        unsafe { Self::from_raw_parts_binding(handle.binding(), strides, shape) }
    }

    pub(crate) unsafe fn from_raw_parts_binding(
        handle: cubecl_runtime::server::Binding,
        strides: Strides,
        shape: Shape,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorBinding::from_raw_parts_binding(handle, strides, shape),
            }
        }
    }

    /// Create an alias argument.
    pub fn into_alias(self, position: usize) -> Self {
        match self {
            TensorArg::Handle { handle } => handle.into_alias(position),
            alias @ TensorArg::Alias { .. } => alias,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            TensorArg::Handle { handle } => handle.size(),
            TensorArg::Alias { shape, .. } => shape.iter().product(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            TensorArg::Handle { handle } => &handle.shape,
            TensorArg::Alias { shape, .. } => shape,
        }
    }

    pub fn strides(&self) -> &[usize] {
        match self {
            TensorArg::Handle { handle } => &handle.strides,
            TensorArg::Alias { strides, .. } => strides,
        }
    }
}

impl<R: Runtime> TensorArg<R> {
    pub fn into_buffer_arg(self) -> BufferArg<R> {
        match self {
            TensorArg::Handle { handle } => {
                let handle = unsafe {
                    let size = handle.size();
                    BufferBinding::from_raw_parts_binding(handle.handle, size)
                };
                BufferArg::Handle { handle }
            }
            TensorArg::Alias {
                input_pos, shape, ..
            } => BufferArg::Alias {
                input_pos,
                length: [shape.iter().product()],
            },
        }
    }
}

impl<R: Runtime> TensorBinding<R> {
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn into_tensor_arg(self) -> TensorArg<R> {
        unsafe { TensorArg::from_raw_parts_binding(self.handle, self.strides, self.shape) }
    }
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn into_alias(self, index: usize) -> TensorArg<R> {
        TensorArg::Alias {
            input_pos: index,
            strides: self.strides,
            shape: self.shape,
        }
    }
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn as_alias(&self, index: usize) -> TensorArg<R> {
        TensorArg::Alias {
            input_pos: index,
            strides: self.strides.clone(),
            shape: self.shape.clone(),
        }
    }
    /// Convert the handle into a [buffer argument](BufferArg).
    pub fn into_buffer_arg(self) -> BufferArg<R> {
        unsafe { BufferArg::from_raw_parts_binding(self.handle, self.shape.iter().product()) }
    }

    /// Create a handle from raw parts.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: cubecl_runtime::server::Handle,
        strides: Strides,
        shape: Shape,
    ) -> Self {
        unsafe { Self::from_raw_parts_binding(handle.binding(), strides, shape) }
    }

    /// Create a handle from raw parts.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_binding(
        handle: cubecl_runtime::server::Binding,
        strides: Strides,
        shape: Shape,
    ) -> Self {
        Self {
            handle,
            strides,
            shape,
            runtime: PhantomData,
        }
    }

    pub fn into_copy_descriptor(self, elem_size: usize) -> CopyDescriptor {
        CopyDescriptor {
            handle: self.handle,
            shape: self.shape,
            strides: self.strides,
            elem_size,
        }
    }
}
