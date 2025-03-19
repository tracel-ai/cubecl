use std::{marker::PhantomData, num::NonZero};

use serde::{Deserialize, Serialize};

use crate::{
    Runtime,
    compute::{KernelBuilder, KernelLauncher},
    ir::{Id, Item, Vectorization},
    prelude::{
        ArgSettings, CompilationArg, CubePrimitive, ExpandElementTyped, LaunchArg, LaunchArgExpand,
        TensorHandleRef,
    },
};

use super::Array;

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ArrayCompilationArg {
    pub inplace: Option<Id>,
    pub vectorisation: Vectorization,
}

impl CompilationArg for ArrayCompilationArg {}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle).
pub struct ArrayHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle,
    pub(crate) length: [usize; 1],
    pub elem_size: usize,
    runtime: PhantomData<R>,
}

impl<C: CubePrimitive> LaunchArgExpand for Array<C> {
    type CompilationArg = ArrayCompilationArg;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Array<C>> {
        builder
            .input_array(Item::vectorized(
                C::as_elem(&builder.context),
                arg.vectorisation,
            ))
            .into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Array<C>> {
        match arg.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder
                .output_array(Item::vectorized(
                    C::as_elem(&builder.context),
                    arg.vectorisation,
                ))
                .into(),
        }
    }
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

impl<R: Runtime> ArgSettings<R> for ArrayArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_array(self)
    }
}

impl<'a, R: Runtime> ArrayArg<'a, R> {
    /// Create a new array argument.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts<E: CubePrimitive>(
        handle: &'a cubecl_runtime::server::Handle,
        length: usize,
        vectorization_factor: u8,
    ) -> Self {
        unsafe {
            ArrayArg::Handle {
                handle: ArrayHandleRef::from_raw_parts(
                    handle,
                    length,
                    E::size().expect("Element should have a size"),
                ),
                vectorization_factor,
            }
        }
    }

    /// Create a new array argument with a manual element size in bytes.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts_and_size(
        handle: &'a cubecl_runtime::server::Handle,
        length: usize,
        vectorization_factor: u8,
        elem_size: usize,
    ) -> Self {
        unsafe {
            ArrayArg::Handle {
                handle: ArrayHandleRef::from_raw_parts(handle, length, elem_size),
                vectorization_factor,
            }
        }
    }
}

impl<'a, R: Runtime> ArrayHandleRef<'a, R> {
    /// Create a new array handle reference.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle,
        length: usize,
        elem_size: usize,
    ) -> Self {
        Self {
            handle,
            length: [length],
            elem_size,
            runtime: PhantomData,
        }
    }

    /// Return the handle as a tensor instead of an array.
    pub fn as_tensor(&self) -> TensorHandleRef<'_, R> {
        let shape = &self.length;

        TensorHandleRef {
            handle: self.handle,
            strides: &[1],
            shape,
            elem_size: self.elem_size,
            runtime: PhantomData,
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Array<C> {
    type RuntimeArg<'a, R: Runtime> = ArrayArg<'a, R>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match runtime_arg {
            ArrayArg::Handle {
                vectorization_factor,
                ..
            } => ArrayCompilationArg {
                inplace: None,
                vectorisation: Vectorization::Some(NonZero::new(*vectorization_factor).unwrap()),
            },
            ArrayArg::Alias { input_pos } => ArrayCompilationArg {
                inplace: Some(*input_pos as Id),
                vectorisation: Vectorization::None,
            },
        }
    }
}
