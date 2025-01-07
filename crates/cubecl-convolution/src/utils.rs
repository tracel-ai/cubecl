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

pub(crate) enum ConvType {
    Conv2d,
    Conv3d,
}

pub(crate) fn bias_reshape_or_zero<R: Runtime, E: Float>(
    client: &ComputeClient<R::Server, R::Channel>,
    bias: Option<TensorHandleRef<R>>,
    shape_out: &[usize],
    conv_type: ConvType,
) -> TensorHandle<R, E> {
    let base_shape = match conv_type {
        ConvType::Conv2d => vec![1, 1, 1],
        ConvType::Conv3d => vec![1, 1, 1, 1],
    };

    match bias {
        Some(bias) => {
            let mut shape = vec![bias.shape[0]];
            shape.extend(&base_shape);
            TensorHandle::new_contiguous(shape, bias.handle.clone())
        }
        None => {
            let mut shape = vec![shape_out[0]];
            shape.extend(&base_shape);
            TensorHandle::zeros(&client.clone(), shape)
        }
    }
}
