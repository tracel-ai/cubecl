use cubecl_core::{
    client::ComputeClient,
    prelude::{Float, TensorHandleRef},
    Runtime,
};
use cubecl_linalg::tensor::TensorHandle;

/// Convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

/// Transposed convolution options.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ConvTransposeOptions<const N: usize> {
    /// Stride (non-zero).
    pub stride: [usize; N],

    /// Padding.
    pub padding: [usize; N],

    /// Padding out.
    pub padding_out: [usize; N],

    /// Dilation (non-zero).
    pub dilation: [usize; N],

    /// Groups (non-zero).
    pub groups: usize,
}

pub(crate) fn bias_reshape_or_zero<R: Runtime, E: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    bias: Option<TensorHandleRef<R>>,
    shape_out: &[usize],
) -> TensorHandle<R, E> {
    match bias {
        Some(bias) => {
            let shape = vec![bias.shape[0], 1, 1, 1, 1];
            TensorHandle::new_contiguous(shape, bias.handle.clone())
        }
        None => {
            let shape = vec![shape_out[0], 1, 1, 1, 1];
            TensorHandle::zeros(&client.clone(), shape)
        }
    }
}
