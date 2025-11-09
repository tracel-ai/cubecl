use std::ops::Deref;

use cubecl_core::CubeDim;
use cubecl_matmul::components::{
    MatmulIdent, MatmulLineSizes, MatmulSetupError, MatrixLayout, TilingScheme,
    global::{
        GlobalConfig, PlaneRoleConfig, SpecializedLoadingSides, multi_stage::EventLoadingMode,
        read::ReaderMode,
    },
    stage::{StageConfig, StageMemoryConfig, SwizzleMode},
};

use super::*;

/// Convolution specific config, extends regular matmul [`Config`](global::Config)
pub trait ConvGemmConfig: GlobalConfig {
    /// The size of the convolution kernel at `dim`
    fn convolution_params(&self) -> ConvolutionParams;
    fn line_sizes(&self) -> MatmulLineSizes;
    fn check_spatial_bounds(&self) -> bool;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionConfig<M: GlobalConfig> {
    matmul: M,
    params: ConvolutionParams,
    num_stages: u32,
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

impl<M: GlobalConfig> GlobalConfig for ConvolutionConfig<M> {
    type StageConfig = M::StageConfig;

    fn stage_memory_config(&self, ident: MatmulIdent) -> StageMemoryConfig {
        self.stage_config().stage_memory_config(ident.into_stage())
    }

    fn stage_config(&self) -> Self::StageConfig {
        self.matmul.stage_config()
    }

    fn global_line_size(&self, ident: MatmulIdent) -> u32 {
        self.matmul.global_line_size(ident)
    }

    fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout {
        self.matmul.matrix_layout(ident)
    }

    fn swizzle_mode(&self, ident: MatmulIdent) -> SwizzleMode {
        self.matmul.swizzle_mode(ident)
    }

    fn num_loading_planes(&self, ident: MatmulIdent) -> u32 {
        self.matmul.num_loading_planes(ident)
    }

    fn plane_dim(&self) -> u32 {
        self.matmul.plane_dim()
    }

    fn check_row_bounds(&self, ident: MatmulIdent) -> bool {
        self.matmul.check_row_bounds(ident)
    }

    fn check_col_bounds(&self, ident: MatmulIdent) -> bool {
        self.matmul.check_col_bounds(ident)
    }

    fn check_k_bounds(&self) -> bool {
        self.matmul.check_k_bounds()
    }

    fn precompute_job(&self) -> bool {
        self.matmul.precompute_job()
    }

    fn num_stages(&self, _ident: MatmulIdent) -> u32 {
        self.num_stages
    }

    fn reader_mode(&self) -> ReaderMode {
        self.matmul.reader_mode()
    }

    fn tiling_scheme(&self) -> TilingScheme {
        self.matmul.tiling_scheme()
    }

    fn event_loading_mode(&self, ident: MatmulIdent) -> EventLoadingMode {
        self.matmul.event_loading_mode(ident)
    }

    fn plane_role_config(&self) -> PlaneRoleConfig {
        self.matmul.plane_role_config()
    }

    fn specialized_loading_sides(&self) -> SpecializedLoadingSides {
        self.matmul.specialized_loading_sides()
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new(self.plane_dim(), self.tiling_scheme().tiles_in_stage_m(), 1)
    }
}

impl<M: GlobalConfig> ConvGemmConfig for ConvolutionConfig<M> {
    fn convolution_params(&self) -> ConvolutionParams {
        self.params
    }

    fn line_sizes(&self) -> cubecl_matmul::components::MatmulLineSizes {
        MatmulLineSizes {
            lhs: self.global_line_size(MatmulIdent::Lhs) as u8,
            rhs: self.global_line_size(MatmulIdent::Rhs) as u8,
            out: self.global_line_size(MatmulIdent::Out) as u8,
        }
    }

    fn check_spatial_bounds(&self) -> bool {
        let spatial_dims = self.params.dimensionality.num_dims();
        let mut has_padding = false;
        for i in 0..spatial_dims {
            has_padding |= self.params.padding[i as usize] != 0;
        }
        has_padding
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
            params,
            num_stages,
        })
    }

    pub fn to_matmul_config(self) -> M {
        self.matmul
    }
}
