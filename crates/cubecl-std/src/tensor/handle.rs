use core::marker::PhantomData;
use cubecl_core::tensor_line_size_parallel;
use cubecl_core::{Runtime, server};
use cubecl_core::{calculate_cube_count_elemwise, server::Allocation};
use cubecl_core::{prelude::*, server::CopyDescriptor};
use cubecl_runtime::server::Handle;
use cubecl_runtime::stride as stride_util;

/// Tensor representation containing a [server handle](Handle) as well as basic tensor metadata.,
pub struct TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    /// The buffer where the data are stored.
    pub handle: server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    elem: PhantomData<E>,
    runtime: PhantomData<R>,
}

impl<R, E> core::fmt::Debug for TensorHandle<R, E>
where
    R: Runtime,
    E: CubePrimitive,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "Tensor {{ shape: {:?}, strides: {:?}, dtype: {}}}",
            self.shape,
            self.strides,
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
    pub fn new(handle: server::Handle, shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            handle,
            shape,
            strides,
            elem: PhantomData,
            runtime: PhantomData,
        }
    }

    pub fn empty(client: &ComputeClient<R::Server, R::Channel>, shape: Vec<usize>) -> Self {
        let elem_size = E::size().expect("To be a native type");
        let Allocation { handle, strides } = client.empty_tensor(&shape, elem_size);

        Self::new(handle, shape, strides)
    }

    /// Create a new tensor.
    pub fn from_ref(handle: &TensorHandleRef<'_, R>) -> Self {
        Self {
            handle: handle.handle.clone(),
            shape: handle.shape.to_vec(),
            strides: handle.strides.to_vec(),
            elem: PhantomData,
            runtime: PhantomData,
        }
    }

    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(shape: Vec<usize>, handle: Handle) -> Self {
        let strides = stride_util::contiguous_strides(&shape);

        Self {
            handle,
            shape,
            strides,
            elem: PhantomData,
            runtime: PhantomData,
        }
    }

    /// Check if the tensor is safe to mutate.
    pub fn can_mut(&self) -> bool {
        self.handle.can_mut()
    }

    pub fn as_ref(&self) -> TensorHandleRef<'_, R> {
        unsafe {
            TensorHandleRef::from_raw_parts(
                &self.handle,
                &self.strides,
                &self.shape,
                size_of::<E>(),
            )
        }
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

    pub fn as_copy_descriptor<'a>(&'a self) -> CopyDescriptor<'a> {
        CopyDescriptor {
            binding: self.handle.clone().binding(),
            shape: &self.shape,
            strides: &self.strides,
            elem_size: size_of::<E>(),
        }
    }
}
impl<R, E> TensorHandle<R, E>
where
    R: Runtime,
    E: Numeric,
{
    pub fn zeros(client: &ComputeClient<R::Server, R::Channel>, shape: Vec<usize>) -> Self {
        let num_elements: usize = shape.iter().product();
        let rank = shape.len();
        let output = Self::empty(client, shape);

        let line_size = tensor_line_size_parallel(
            R::supported_line_sizes().iter().cloned(),
            &output.shape,
            &output.strides,
            rank - 1,
        );

        let cube_dim = CubeDim::default();
        let cube_count = calculate_cube_count_elemwise(num_elements / line_size as usize, cube_dim);
        let array_len = output.handle.size() as usize / size_of::<E>();

        unsafe {
            init::zeros_array::launch_unchecked::<E, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<E>(&output.handle, array_len, line_size),
            )
        };

        output
    }
}

pub(crate) mod init {
    use cubecl::prelude::*;
    use cubecl_core as cubecl;

    #[cube(launch_unchecked)]
    pub fn zeros_array<C: Numeric>(output: &mut Array<Line<C>>) {
        if ABSOLUTE_POS < output.len() {
            output[ABSOLUTE_POS] = Line::cast_from(C::from_int(0));
        }
    }
}
