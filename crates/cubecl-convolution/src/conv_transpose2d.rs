use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::ConvTransposeOptions;

/// The strategy to be used when launching a conv_transpose kernel.
pub enum ConvTranspose2dStrategy {
    /// A simple direct convolution.
    Direct,
    /// GEMM (im2col) based implementation of convolution. Significantly increased memory usage.
    Gemm,
}

impl Default for ConvTranspose2dStrategy {
    fn default() -> Self {
        // Default to the more memory-conservative algorithm
        ConvTranspose2dStrategy::Direct
    }
}

/// Perform a 2D convolution with the given strategy
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
/// * `strategy` - The convolution algorithm to use. Autotune will pick the fastest available option.
///
pub fn conv_transpose2d<R: Runtime, E: Float>(
    input: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: Option<TensorHandleRef<R>>,
    out: TensorHandleRef<R>,
    options: ConvTransposeOptions<2>,
    strategy: ConvTranspose2dStrategy,
) {
    match strategy {
        ConvTranspose2dStrategy::Direct => {
            // conv_transpose2d_direct::<R, E>(input, weight, bias, options)
        }
        ConvTranspose2dStrategy::Gemm => {
            // conv_transpose2d_col2im::<R, E>(input, weight, bias, options)
        }
    }
}
