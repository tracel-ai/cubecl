use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::ConvOptions;

/// The strategy to be used when launching a convolution kernel.
pub enum Conv2dStrategy {
    /// A simple direct convolution.
    Direct,
    /// GEMM (im2col) based implementation of convolution. Significantly increased memory usage.
    Gemm,
    /// Implicit GEMM implementation of convolution. Lower memory usage but requires CMMA and
    /// has constraints on tensor shape.
    ImplicitGemm,
    /// Implicit GEMM implementation of convolution. Uses `cubecl` matmul components to provide
    /// the flexibility needed to work well for varied problem sizes.
    ImplicitGemmComplex,
}

impl Default for Conv2dStrategy {
    fn default() -> Self {
        Conv2dStrategy::Direct
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
pub fn conv2d<R: Runtime>(
    input: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: Option<TensorHandleRef<R>>,
    out: TensorHandleRef<R>,
    options: ConvOptions<2>,
    strategy: Conv2dStrategy,
) {
    match strategy {
        Conv2dStrategy::Direct => {
            // conv2d_direct::<R, E>(input, weight, bias, options)
        }
        Conv2dStrategy::Gemm => {
            //conv2d_im2col::<R, E>(input, weight, bias, options)
        }
        Conv2dStrategy::ImplicitGemm => { //conv2d_implicit_gemm::<R, E>(input, weight, bias, options)
        }
        Conv2dStrategy::ImplicitGemmComplex => {
            // conv2d_gemm_cmma_large_m::<R, E>(input, weight, bias, options)
        }
    }
}
