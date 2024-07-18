use cubecl_core::prelude::*;
use cubecl_core::Runtime;
use cubecl_runtime::server::Handle;
use std::marker::PhantomData;

use super::layout::{memory_layout, MatrixLayout};

/// Tensor representation containing a [server handle](Handle) as well as basic tensor metadata.,
pub struct TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    /// The buffer where the data are stored.
    pub handle: Handle<R::Server>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The strides of the tensor.
    pub strides: Vec<usize>,
    elem: PhantomData<E>,
}

impl<R, E> core::fmt::Debug for TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Tensor {{ shape: {:?}, strides: {:?}, runtime: {}, dtype: {}}}",
            self.shape,
            self.strides,
            R::name(),
            core::any::type_name::<E>(),
        ))
    }
}

impl<R, E> Clone for TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            elem: PhantomData,
        }
    }
}

impl<R, E> TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(shape: Vec<usize>, handle: Handle<R::Server>) -> Self {
        let d = shape.len();
        let mut strides = Vec::with_capacity(d);

        let mut current = 1;
        shape.iter().enumerate().rev().for_each(|(_, val)| {
            strides.push(current);
            current *= val;
        });
        strides.reverse();

        Self {
            handle,
            shape,
            strides,
            elem: PhantomData,
        }
    }

    /// Check if the tensor is safe to mutate.
    pub fn can_mut(&self) -> bool {
        self.handle.can_mut()
    }

    /// Check if the current tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.matrix_layout() == MatrixLayout::Contiguous
    }

    pub(crate) fn matrix_layout(&self) -> MatrixLayout {
        memory_layout(&self.strides)
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }
}
