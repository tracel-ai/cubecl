use core::marker::PhantomData;
use cubecl_core::{Runtime, server, zspace::strides};
use cubecl_core::{calculate_cube_count_elemwise, server::MemoryLayout};
use cubecl_core::{ir::StorageType, zspace::metadata::Metadata};
use cubecl_core::{prelude::*, server::CopyDescriptor};
use cubecl_core::{
    tensor_vector_size_parallel,
    zspace::{Shape, Strides},
};
use cubecl_runtime::server::Handle;

/// Tensor representation containing a [server handle](Handle) as well as basic tensor metadata.,
pub struct TensorHandle<R>
where
    R: Runtime,
{
    /// The buffer where the data are stored.
    pub handle: server::Handle,
    pub metadata: Box<Metadata>,
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
            self.shape(),
            self.strides(),
            self.dtype,
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
            metadata: self.metadata.clone(),
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
        shape: impl Into<Shape>,
        strides: impl Into<Strides>,
        storage: impl Into<Type>,
    ) -> Self {
        Self {
            handle,
            metadata: Box::new(Metadata::new(shape, strides)),
            dtype: storage.into().storage_type(),
            runtime: PhantomData,
        }
    }

    pub fn empty(
        client: &ComputeClient<R>,
        shape: impl Into<Shape>,
        storage: impl Into<Type>,
    ) -> Self {
        let storage = storage.into();
        let shape: Shape = shape.into();
        let elem_size = storage.storage_type().size();
        let MemoryLayout {
            memory: handle,
            strides,
        } = client.empty_tensor(shape.clone(), elem_size);

        Self::new(handle, shape, strides, storage)
    }

    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(shape: impl Into<Shape>, handle: Handle, storage: StorageType) -> Self {
        let shape = shape.into();
        let strides = Self::contiguous_strides(&shape);

        Self {
            handle,
            metadata: Box::new(Metadata::new(shape, strides)),
            dtype: storage,
            runtime: PhantomData,
        }
    }

    /// Check if the tensor is safe to mutate.
    pub fn can_mut(&self) -> bool {
        self.handle.can_mut()
    }

    pub fn binding(self) -> TensorBinding<R> {
        unsafe {
            TensorBinding::from_raw_parts(self.handle, self.metadata.strides, self.metadata.shape)
        }
    }

    /// Return the reference to a tensor argument.
    pub fn into_arg(self) -> TensorArg<R> {
        self.binding().into_tensor_arg()
    }

    pub fn into_copy_descriptor(self) -> CopyDescriptor {
        CopyDescriptor {
            handle: self.handle.binding(),
            shape: self.metadata.shape,
            strides: self.metadata.strides,
            elem_size: self.dtype.size(),
        }
    }

    pub fn required_address_type(&self) -> AddressType {
        let len = self.handle.size() / self.dtype.size() as u64;
        AddressType::from_len(len as usize)
    }

    pub fn shape(&self) -> &Shape {
        self.metadata.shape()
    }

    pub fn strides(&self) -> &Strides {
        self.metadata.strides()
    }

    fn contiguous_strides(shape: &[usize]) -> Strides {
        let mut strides = strides![1; shape.len()];

        let mut current = 1;
        shape.iter().rev().enumerate().for_each(|(i, val)| {
            strides[i] = current;
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
    pub fn zeros(
        client: &ComputeClient<R>,
        shape: impl Into<Shape>,
        dtype: impl Into<Type>,
    ) -> Self {
        let dtype = dtype.into();
        let shape = shape.into();
        let num_elements: usize = shape.iter().product();
        let rank = shape.len();
        let output = Self::empty(client, shape, dtype);
        let dtype = dtype.storage_type();

        let vector_size = tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(dtype.size()),
            output.shape(),
            output.strides(),
            rank - 1,
        );

        let working_units = num_elements / vector_size as usize;
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
        let array_len = output.handle.size_in_used() as usize / dtype.size();

        unsafe {
            init::zeros_array::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                output.required_address_type(),
                vector_size,
                BufferArg::from_raw_parts(output.handle.clone(), array_len),
                dtype,
            )
        };

        output
    }
}

pub(crate) mod init {
    use cubecl::prelude::*;
    use cubecl_core::{self as cubecl, ir::StorageType};

    #[cube(launch_unchecked, address_type = "dynamic")]
    pub fn zeros_array<C: Numeric, N: Size>(
        output: &mut [Vector<C, N>],
        #[define(C)] _elem: StorageType,
    ) {
        if ABSOLUTE_POS < output.len() {
            output[ABSOLUTE_POS] = Vector::cast_from(C::from_int(0));
        }
    }
}
