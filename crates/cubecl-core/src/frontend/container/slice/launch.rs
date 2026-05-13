use alloc::boxed::Box;
use core::marker::PhantomData;

use crate::{frontend::container::slice, prelude::*};
use cubecl_ir::Id;
use cubecl_runtime::runtime::Runtime;
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct BufferCompilationArg {
    pub inplace: Option<Id>,
}

/// Buffer representation with a reference to the [server handle](cubecl_runtime::server::Handle).
pub struct BufferBinding<R: Runtime> {
    pub handle: cubecl_runtime::server::Binding,
    pub(crate) length: [usize; 1],
    runtime: PhantomData<R>,
}

pub enum BufferArg<R: Runtime> {
    /// The buffer is passed with a buffer handle.
    Handle {
        /// The buffer handle.
        handle: BufferBinding<R>,
    },
    /// The buffer is aliasing another input buffer.
    Alias {
        /// The position of the input buffer.
        input_pos: usize,
        /// The length of the underlying handle
        length: [usize; 1],
    },
}

impl<R: Runtime> BufferArg<R> {
    /// Create a new buffer argument.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe {
            BufferArg::Handle {
                handle: BufferBinding::from_raw_parts(handle, length),
            }
        }
    }
    /// Create a new buffer argument from a binding.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_binding(
        binding: cubecl_runtime::server::Binding,
        length: usize,
    ) -> Self {
        unsafe {
            BufferArg::Handle {
                handle: BufferBinding::from_raw_parts_binding(binding, length),
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            BufferArg::Handle { handle } => handle.length[0],
            BufferArg::Alias { length, .. } => length[0],
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            BufferArg::Handle { handle } => &handle.length,
            BufferArg::Alias { length, .. } => length,
        }
    }
}

impl<R: Runtime> BufferBinding<R> {
    /// Create a new buffer handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe { Self::from_raw_parts_binding(handle.binding(), length) }
    }

    /// Create a new buffer handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length or size, may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_binding(
        handle: cubecl_runtime::server::Binding,
        length: usize,
    ) -> Self {
        Self {
            handle,
            length: [length],
            runtime: PhantomData,
        }
    }

    /// Return the handle as a tensor instead of a buffer.
    pub fn into_tensor(self) -> TensorBinding<R> {
        let shape = self.length.into();

        TensorBinding {
            handle: self.handle,
            strides: [1].into(),
            shape,
            runtime: PhantomData,
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Box<[C]> {
    type RuntimeArg<R: Runtime> = BufferArg<R>;
    type CompilationArg = BufferCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        <[C]>::register(arg, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<Box<[C]>> {
        <[C]>::expand(arg, builder).expand.into()
    }
}

impl<C: CubePrimitive> LaunchArg for [C] {
    type RuntimeArg<R: Runtime> = BufferArg<R>;
    type CompilationArg = BufferCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let ty = launcher.with_scope(|scope| C::__expand_as_type(scope));
        let inplace = match &arg {
            BufferArg::Handle { .. } => None,
            BufferArg::Alias { input_pos, .. } => Some(*input_pos as Id),
        };
        launcher.register_buffer(arg, ty);

        BufferCompilationArg { inplace }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<[C]> {
        let buffer = match arg.inplace {
            Some(id) => builder.inplace(id),
            None => builder.buffer(C::__expand_as_type(&builder.scope)),
        };
        let scope = &builder.scope;
        let len = expand_buffer_length_native(scope, buffer);
        let slice_var =
            slice::from_raw_parts::<C>(scope, buffer, 0usize.into_expand(scope), len.into());
        slice_var.expand.into()
    }
}
