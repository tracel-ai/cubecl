use std::marker::PhantomData;

use cubecl::{prelude::*, server::Handle, std::tensor::compact_strides};

/// Simple GpuTensor
#[derive(Debug)]
pub struct GpuTensor<R: Runtime, F: Float + CubeElement> {
    data: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: Float + CubeElement> Clone for GpuTensor<R, F> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(), // Handle is a pointer to the data, so cloning it is cheap
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }
}

impl<R: Runtime, F: Float + CubeElement> GpuTensor<R, F> {
    /// Create a GpuTensor with a shape filled by number in order
    pub fn arange(shape: Vec<usize>, client: &ComputeClient<R::Server, R::Channel>) -> Self {
        let size = shape.iter().product();
        let data: Vec<F> = (0..size).map(|i| F::from_int(i as i64)).collect();
        let data = client.create(F::as_bytes(&data));

        let strides = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Create an empty GpuTensor with a shape
    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R::Server, R::Channel>) -> Self {
        let size = shape.iter().product();
        let data = client.empty(size);

        let strides = compact_strides(&shape);
        Self {
            data,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Create a TensorArg to pass to a kernel
    pub fn into_tensor_arg(&self, line_size: u8) -> TensorArg<'_, R> {
        unsafe { TensorArg::from_raw_parts::<F>(&self.data, &self.strides, &self.shape, line_size) }
    }

    /// Return the data from the client
    pub fn read(self, client: &ComputeClient<R::Server, R::Channel>) -> Vec<F> {
        let bytes = client.read_one(self.data.binding());
        F::from_bytes(&bytes).to_vec()
    }
}
