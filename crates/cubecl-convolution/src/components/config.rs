use std::ops::Deref;

use cubecl_core::CubeDim;
use cubecl_matmul::components::{
    MatmulLineSizes, MatmulSetupError,
    global::{GlobalConfig, memory::GlobalMemoryConfig},
};
use std::fmt::Debug;
use std::hash::Hash;

use super::*;

/// Convolution specific config, extends regular matmul [`Config`](global::Config)
pub trait ConvGemmConfig:
    Copy + Clone + Eq + PartialEq + Hash + Debug + Send + Sync + 'static
{
    type GlobalMatmulConfig: GlobalConfig;

    fn matmul_config(&self) -> Self::GlobalMatmulConfig;

    /// The size of the convolution kernel at `dim`
    fn convolution_params(&self) -> ConvolutionParams;
    fn line_sizes(&self) -> MatmulLineSizes;
    fn check_spatial_bounds(&self) -> bool;
    fn cube_dim(&self) -> CubeDim;
    fn lhs_global_memory_config(&self) -> GlobalMemoryConfig;
    fn rhs_global_memory_config(&self) -> GlobalMemoryConfig;
    fn out_global_memory_config(&self) -> GlobalMemoryConfig;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionConfig<M: GlobalConfig> {
    pub matmul: M,
    pub convolution_params: ConvolutionParams,
    pub num_stages: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionParams {
    pub kernel_size: [u32; 3],
    pub stride: [u32; 3],
    pub dilation: [u32; 3],
    pub padding: [i32; 3],
    pub dimensionality: Dimensionality,
}

impl<M: GlobalConfig> Deref for ConvolutionConfig<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.matmul
    }
}

impl<M: GlobalConfig> ConvGemmConfig for ConvolutionConfig<M> {
    type GlobalMatmulConfig = M;

    fn matmul_config(&self) -> Self::GlobalMatmulConfig {
        self.matmul
    }

    fn line_sizes(&self) -> cubecl_matmul::components::MatmulLineSizes {
        self.matmul.global_line_sizes()
    }

    fn cube_dim(&self) -> CubeDim {
        self.matmul.cube_dim()
    }

    fn check_spatial_bounds(&self) -> bool {
        let spatial_dims = self.convolution_params.dimensionality.num_dims();
        let mut has_padding = false;
        for i in 0..spatial_dims {
            has_padding |= self.convolution_params.padding[i as usize] != 0;
        }
        has_padding
    }

    fn convolution_params(&self) -> ConvolutionParams {
        self.convolution_params
    }

    fn lhs_global_memory_config(&self) -> GlobalMemoryConfig {
        self.matmul.lhs_reader_config().gmem_config
    }

    fn rhs_global_memory_config(&self) -> GlobalMemoryConfig {
        self.matmul.rhs_reader_config().gmem_config
    }

    fn out_global_memory_config(&self) -> GlobalMemoryConfig {
        self.matmul.writer_config().gmem_config
    }
}

impl<M: GlobalConfig> ConvolutionConfig<M> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        matmul: M,
        kernel_size: &[u32],
        stride: &[u32],
        dilation: &[u32],
        padding: &[i32],
        dim: Dimensionality,
        num_stages: u32,
    ) -> Result<Self, MatmulSetupError> {
        let dims = kernel_size.len();

        let mut params = ConvolutionParams {
            kernel_size: [0; 3],
            stride: [0; 3],
            dilation: [0; 3],
            padding: [0; 3],
            dimensionality: dim,
        };
        params.kernel_size[0..dims].copy_from_slice(kernel_size);
        params.stride[0..dims].copy_from_slice(stride);
        params.dilation[0..dims].copy_from_slice(dilation);
        params.padding[0..dims].copy_from_slice(padding);
        Ok(Self {
            matmul,
            convolution_params: params,
            num_stages,
        })
    }
}
