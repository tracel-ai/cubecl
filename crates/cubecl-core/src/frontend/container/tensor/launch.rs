use core::marker::PhantomData;

use cubecl_ir::AddressType;
use cubecl_runtime::{runtime::Runtime, server::CopyDescriptor};
use cubecl_zspace::{Shape, Strides};
use serde::{Deserialize, Serialize};

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::Id,
    prelude::{ArrayArg, ArrayBinding, CubePrimitive, LaunchArg, NativeExpand},
};

use super::Tensor;

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
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TensorCompilationArg {
    pub inplace: Option<Id>,
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<R: Runtime> = TensorArg<R>;
    type CompilationArg = TensorCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let ty = launcher.with_scope(|scope| C::as_type(scope));
        let compilation_arg = match &arg {
            TensorArg::Handle { .. } => TensorCompilationArg { inplace: None },
            TensorArg::Alias { input_pos, .. } => TensorCompilationArg {
                inplace: Some(*input_pos as Id),
            },
        };
        launcher.register_tensor(arg, ty);
        compilation_arg
    }

    fn expand(_arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<Tensor<C>> {
        builder.input_tensor(C::as_type(&builder.scope)).into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> NativeExpand<Tensor<C>> {
        match arg.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder.output_tensor(C::as_type(&builder.scope)).into(),
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
    pub fn into_array_arg(self) -> ArrayArg<R> {
        match self {
            TensorArg::Handle { handle } => {
                let handle = unsafe {
                    let size = handle.size();
                    ArrayBinding::from_raw_parts_binding(handle.handle, size)
                };
                ArrayArg::Handle { handle }
            }
            TensorArg::Alias {
                input_pos, shape, ..
            } => ArrayArg::Alias {
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
    /// Convert the handle into an [array argument](ArrayArg).
    pub fn into_array_arg(self) -> ArrayArg<R> {
        let length = self.shape.iter().product();
        unsafe { ArrayArg::from_raw_parts_binding(self.handle, length) }
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
