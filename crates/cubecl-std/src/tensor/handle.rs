use core::marker::PhantomData;
use cubecl_core::ir::StorageType;
use cubecl_core::tensor_line_size_parallel;
use cubecl_core::{Runtime, server};
use cubecl_core::{calculate_cube_count_elemwise, server::Allocation};
use cubecl_core::{prelude::*, server::CopyDescriptor};
use cubecl_runtime::server::Handle;

/// Tensor representation containing a [server handle](Handle) as well as basic tensor metadata.,
pub struct TensorHandle<R>
where
    R: Runtime,
{
    /// The buffer where the data are stored.
    pub handle: server::Handle,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    /// The type used as storage.
    pub dtype: StorageType,
    runtime: PhantomData<R>,
}

impl<R> core::fmt::Debug for TensorHandle<R>
where
    R: Runtime,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "Tensor {{ shape: {:?}, strides: {:?}, dtype: {}}}",
            self.shape, self.strides, self.dtype,
        ))
    }
}

impl<R> Clone for TensorHandle<R>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            runtime: PhantomData,
        }
    }
}

impl<R> TensorHandle<R>
where
    R: Runtime,
{
    /// Create a new tensor.
    pub fn new(
        handle: server::Handle,
        shape: Vec<usize>,
        strides: Vec<usize>,
        storage: StorageType,
    ) -> Self {
        Self {
            handle,
            shape,
            strides,
            dtype: storage,
            runtime: PhantomData,
        }
    }

    pub fn empty(client: &ComputeClient<R>, shape: Vec<usize>, storage: StorageType) -> Self {
        let elem_size = storage.size();
        let Allocation { handle, strides } = client.empty_tensor(&shape, elem_size);

        Self::new(handle, shape, strides, storage)
    }

    /// Create a new tensor.
    pub fn from_ref(handle: &TensorHandleRef<'_, R>, storage: StorageType) -> Self {
        Self {
            handle: handle.handle.clone(),
            shape: handle.shape.to_vec(),
            strides: handle.strides.to_vec(),
            dtype: storage,
            runtime: PhantomData,
        }
    }

    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(shape: Vec<usize>, handle: Handle, storage: StorageType) -> Self {
        let strides = Self::contiguous_strides(&shape);

        Self {
            handle,
            shape,
            strides,
            dtype: storage,
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
                self.dtype.size(),
            )
        }
    }

    /// Return the reference to a tensor argument.
    pub fn as_arg<'a>(&'a self, line_size: LineSize) -> TensorArg<'a, R> {
        let handle: TensorHandleRef<'a, R> = self.as_ref();

        unsafe {
            TensorArg::from_raw_parts_and_size(
                handle.handle,
                handle.strides,
                handle.shape,
                line_size,
                handle.elem_size,
            )
        }
    }

    pub fn as_copy_descriptor<'a>(&'a self) -> CopyDescriptor<'a> {
        CopyDescriptor {
            binding: self.handle.clone().binding(),
            shape: &self.shape,
            strides: &self.strides,
            elem_size: self.dtype.size(),
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
}
impl<R> TensorHandle<R>
where
    R: Runtime,
{
    pub fn zeros(client: &ComputeClient<R>, shape: Vec<usize>, dtype: StorageType) -> Self {
        let num_elements: usize = shape.iter().product();
        let rank = shape.len();
        let output = Self::empty(client, shape, dtype);

        let line_size = tensor_line_size_parallel(
            R::supported_line_sizes().iter().cloned(),
            &output.shape,
            &output.strides,
            rank - 1,
        );

        let working_units = num_elements / line_size as usize;
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
        let array_len = output.handle.size() as usize / dtype.size();

        unsafe {
            init::zeros_array::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts_and_size(
                    &output.handle,
                    array_len,
                    line_size,
                    dtype.size(),
                ),
                dtype,
            )
            .expect("Should be able to launch the kernel all the time")
        };

        output
    }
}

pub(crate) mod init {
    use cubecl::prelude::*;
    use cubecl_core::{self as cubecl, ir::StorageType};

    #[cube(launch_unchecked)]
    pub fn zeros_array<C: Numeric>(output: &mut Array<Line<C>>, #[define(C)] _elem: StorageType) {
        if ABSOLUTE_POS < output.len() {
            output[ABSOLUTE_POS] = Line::cast_from(C::from_int(0));
        }
    }
}
