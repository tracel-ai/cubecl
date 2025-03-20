use cubecl_core::calculate_cube_count_elemwise;
use cubecl_core::prelude::*;
use cubecl_core::tensor_line_size_parallel;
use cubecl_core::{Runtime, server};
use cubecl_runtime::server::Handle;
use std::marker::PhantomData;

/// Tensor representation containing a [server handle](Handle) as well as basic tensor metadata.,
pub struct TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    /// The buffer where the data are stored.
    pub handle: server::TensorHandle,
    elem: PhantomData<E>,
    runtime: PhantomData<R>,
}

impl<R, E> core::fmt::Debug for TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Tensor {{ shape: {:?}, strides: {:?}, dtype: {}}}",
            self.handle.shape,
            self.handle.strides,
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
            elem: PhantomData,
            runtime: PhantomData,
        }
    }
}

impl<R, E> TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    /// Create a new tensor.
    pub fn new(handle: server::TensorHandle) -> Self {
        Self {
            handle,
            elem: PhantomData,
            runtime: PhantomData,
        }
    }

    /// Create a new tensor.
    pub fn from_ref(handle: &TensorHandleRef<'_, R>) -> Self {
        let handle = server::TensorHandle::new(
            handle.handle.clone(),
            handle.strides.to_vec(),
            handle.shape.to_vec(),
            handle.elem_size,
        );
        Self {
            handle,
            elem: PhantomData,
            runtime: PhantomData,
        }
    }

    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(shape: Vec<usize>, elem_size: usize, handle: Handle) -> Self {
        let strides = Self::contiguous_strides(&shape);
        let handle = server::TensorHandle::new(handle, strides, shape, elem_size);

        Self {
            handle,
            elem: PhantomData,
            runtime: PhantomData,
        }
    }

    /// Check if the tensor is safe to mutate.
    pub fn can_mut(&self) -> bool {
        self.handle.can_mut()
    }

    pub fn as_ref(&self) -> TensorHandleRef<'_, R> {
        TensorHandleRef::from_handle(&self.handle)
    }

    /// Return the reference to a tensor argument.
    pub fn as_arg<'a>(&'a self, vectorisation: u8) -> TensorArg<'a, R> {
        let handle: TensorHandleRef<'a, R> = self.as_ref();

        unsafe {
            TensorArg::from_raw_parts::<E>(
                handle.handle,
                handle.strides,
                handle.shape,
                vectorisation,
            )
        }
    }

    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());

        let mut current = 1;
        shape.iter().enumerate().rev().for_each(|(_, val)| {
            strides.push(current);
            current *= val;
        });
        strides.reverse();
        strides
    }

    pub fn shape(&self) -> &[usize] {
        &self.handle.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.handle.strides
    }

    pub fn shape_mut(&mut self) -> &mut Vec<usize> {
        &mut self.handle.shape
    }

    pub fn strides_mut(&mut self) -> &mut Vec<usize> {
        &mut self.handle.strides
    }
}
impl<R, E> TensorHandle<R, E>
where
    R: Runtime,
    E: Numeric,
{
    pub fn empty(client: &ComputeClient<R::Server, R::Channel>, shape: Vec<usize>) -> Self {
        let elem_size = E::size().expect("To be a native type");
        let handle = client.empty_tensor(shape, elem_size);

        Self::new(handle)
    }

    pub fn zeros(client: &ComputeClient<R::Server, R::Channel>, shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let rank = shape.len();
        let output = Self::empty(client, shape);

        let vectorization_factor = tensor_line_size_parallel(
            R::supported_line_sizes().iter().cloned(),
            &output.handle.shape,
            &output.handle.strides,
            rank - 1,
        );

        let cube_dim = CubeDim::default();
        let cube_count =
            calculate_cube_count_elemwise(num_elements / vectorization_factor as usize, cube_dim);
        let array_len = output.handle.handle.size();

        unsafe {
            init::zeros_array::launch_unchecked::<E, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<E>(
                    &output.handle.handle,
                    array_len as usize,
                    vectorization_factor,
                ),
            )
        };

        output
    }
}

pub(crate) mod init {
    use cubecl::prelude::*;
    use cubecl_core as cubecl;

    #[cube(launch_unchecked)]
    pub fn zeros_array<C: Numeric>(output: &mut Array<C>) {
        if ABSOLUTE_POS < output.len() {
            output[ABSOLUTE_POS] = C::from_int(0);
        }
    }
}
