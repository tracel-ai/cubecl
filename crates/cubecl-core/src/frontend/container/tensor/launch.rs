use core::marker::PhantomData;

use cubecl_ir::AddressType;
use cubecl_runtime::{runtime::Runtime, server::CopyDescriptor};
use cubecl_zspace::{Shape, Strides};
use serde::{Deserialize, Serialize};

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::{Id, LineSize, Type},
    prelude::{
        ArgSettings, ArrayArg, CompilationArg, CubePrimitive, ExpandElementTyped, LaunchArg,
    },
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
        line_size: LineSize,
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
    pub handle: &'a cubecl_runtime::server::HandleBinding,
    pub strides: Strides,
    pub shape: Shape,
    pub elem_size: usize,
    pub runtime: PhantomData<R>,
}

impl<'a, R: Runtime> Clone for TensorHandleRef<'a, R> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
            strides: self.strides.clone(),
            shape: self.shape.clone(),
            elem_size: self.elem_size,
            runtime: PhantomData,
        }
    }
}

impl<R: Runtime> TensorHandleRef<'_, R> {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Address type required to fully index this tensor handle, assuming scalar access.
    pub fn required_address_type(&self) -> AddressType {
        AddressType::from_len(self.handle.size() as usize / self.elem_size)
    }
}

impl<R: Runtime> core::fmt::Debug for TensorHandleRef<'_, R> {
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
    pub line_size: LineSize,
}

impl CompilationArg for TensorCompilationArg {}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;
    type CompilationArg = TensorCompilationArg;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match runtime_arg {
            TensorArg::Handle { line_size, .. } => TensorCompilationArg {
                inplace: None,
                line_size: *line_size as LineSize,
            },
            TensorArg::Alias { input_pos } => TensorCompilationArg {
                inplace: Some(*input_pos as Id),
                line_size: 0,
            },
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        builder
            .input_tensor(Type::new(C::as_type(&builder.scope)).line(arg.line_size))
            .into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        match arg.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder
                .output_tensor(Type::new(C::as_type(&builder.scope)).line(arg.line_size))
                .into(),
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
    pub unsafe fn from_raw_parts<E: CubePrimitive>(
        handle: &'a cubecl_runtime::server::HandleBinding,
        strides: Strides,
        shape: Shape,
        factor: LineSize,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(
                    handle,
                    strides,
                    shape,
                    E::size().expect("Element should have a size"),
                ),
                line_size: factor,
            }
        }
    }

    /// Create a new tensor argument specified with its vectorization factor with a manual element
    /// size in bytes.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts_and_size(
        handle: &'a cubecl_runtime::server::HandleBinding,
        strides: Strides,
        shape: Shape,
        factor: LineSize,
        elem_size: usize,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(handle, strides, shape, elem_size),
                line_size: factor,
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

impl<R: Runtime> ArgSettings<R> for TensorArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor(self);
    }
}

impl<'a, R: Runtime> TensorHandleRef<'a, R> {
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn as_tensor_arg(&'a self, line_size: LineSize) -> TensorArg<'a, R> {
        unsafe {
            TensorArg::from_raw_parts_and_size(
                self.handle,
                self.strides.clone(),
                self.shape.clone(),
                line_size,
                self.elem_size,
            )
        }
    }
    /// Convert the handle into an [array argument](ArrayArg).
    pub fn as_array_arg(&'a self, line_size: LineSize) -> ArrayArg<'a, R> {
        let length = self.shape.iter().product();
        unsafe { ArrayArg::from_raw_parts_and_size(self.handle, length, line_size, self.elem_size) }
    }
    /// Create a handle from raw parts.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::HandleBinding,
        strides: Strides,
        shape: Shape,
        elem_size: usize,
    ) -> Self {
        Self {
            handle,
            strides,
            shape,
            elem_size,
            runtime: PhantomData,
        }
    }

    pub fn as_copy_descriptor(&self) -> CopyDescriptor {
        CopyDescriptor {
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            elem_size: self.elem_size,
        }
    }
}
