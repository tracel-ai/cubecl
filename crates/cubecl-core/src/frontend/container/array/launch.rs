use core::marker::PhantomData;

use cubecl_runtime::runtime::Runtime;
use serde::{Deserialize, Serialize};

use crate::{
    compute::{KernelBuilder, KernelLauncher},
    ir::Id,
    prelude::{CubePrimitive, LaunchArg, NativeExpand, TensorBinding},
};

use super::Array;

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ArrayCompilationArg {
    pub inplace: Option<Id>,
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle).
pub struct ArrayBinding<R: Runtime> {
    pub handle: cubecl_runtime::server::Binding,
    pub(crate) length: [usize; 1],
    runtime: PhantomData<R>,
}

pub enum ArrayArg<R: Runtime> {
    /// The array is passed with an array handle.
    Handle {
        /// The array handle.
        handle: ArrayBinding<R>,
    },
    /// The array is aliasing another input array.
    Alias {
        /// The position of the input array.
        input_pos: usize,
        /// The length of the underlying handle
        length: [usize; 1],
    },
}

impl<R: Runtime> ArrayArg<R> {
    /// Create a new array argument.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe {
            ArrayArg::Handle {
                handle: ArrayBinding::from_raw_parts(handle, length),
            }
        }
    }
    /// Create a new array argument from a binding.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_binding(
        binding: cubecl_runtime::server::Binding,
        length: usize,
    ) -> Self {
        unsafe {
            ArrayArg::Handle {
                handle: ArrayBinding::from_raw_parts_binding(binding, length),
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            ArrayArg::Handle { handle } => handle.length[0],
            ArrayArg::Alias { length, .. } => length[0],
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            ArrayArg::Handle { handle } => &handle.length,
            ArrayArg::Alias { length, .. } => length,
        }
    }
}

impl<R: Runtime> ArrayBinding<R> {
    /// Create a new array handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe { Self::from_raw_parts_binding(handle.binding(), length) }
    }

    /// Create a new array handle reference.
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

    /// Return the handle as a tensor instead of an array.
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

impl<C: CubePrimitive> LaunchArg for Array<C> {
    type RuntimeArg<R: Runtime> = ArrayArg<R>;
    type CompilationArg = ArrayCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let ty = launcher.with_scope(|scope| C::as_type(scope));
        let compilation_arg = match &arg {
            ArrayArg::Handle { .. } => ArrayCompilationArg { inplace: None },
            ArrayArg::Alias { input_pos, .. } => ArrayCompilationArg {
                inplace: Some(*input_pos as Id),
            },
        };
        launcher.register_array(arg, ty);
        compilation_arg
    }

    fn expand(_arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<Array<C>> {
        let ty = C::as_type(&builder.scope);
        builder.input_array(ty).into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> NativeExpand<Array<C>> {
        match arg.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder.output_array(C::as_type(&builder.scope)).into(),
        }
    }
}
