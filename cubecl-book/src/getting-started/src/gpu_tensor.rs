use std::marker::PhantomData;

use cubecl::{prelude::*, server::Handle, std::tensor::compact_strides};

/// A contiguous tensor stored on the device.
#[derive(Debug, Clone)]
pub struct GpuTensor<F: Float + CubeElement> {
    data: Handle,
    shape: Vec<usize>,
    _f: PhantomData<F>,
}

impl<F: Float + CubeElement> GpuTensor<F> {
    /// Create a tensor filled with consecutive values.
    pub fn arange(shape: Vec<usize>, client: &Client) -> Self {
        let size = shape.iter().product();
        let data: Vec<F> = (0..size).map(|i| F::from_int(i as i64)).collect();
        let data = client.create_from_slice(F::as_bytes(&data));
        Self {
            data,
            shape,
            _f: PhantomData,
        }
    }

    /// Allocate an uninitialized tensor that the kernel must fill before reading.
    pub fn empty(shape: Vec<usize>, client: &Client) -> Self {
        let size = shape.iter().product::<usize>() * core::mem::size_of::<F>();
        let data = client.empty(size);
        Self {
            data,
            shape,
            _f: PhantomData,
        }
    }

    /// Pass the tensor's storage, shape, and strides to a kernel.
    pub fn as_arg(&self) -> TensorArg {
        unsafe {
            TensorArg::from_raw_parts(
                self.data.clone(),
                compact_strides(&self.shape),
                self.shape.clone().into(),
            )
        }
    }

    /// Read the tensor back from the device.
    pub fn read(self, client: &Client) -> Vec<F> {
        let bytes = client.read_one(self.data).expect("Failed to read tensor");
        F::from_bytes(&bytes).to_vec()
    }
}
