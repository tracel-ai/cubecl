use std::ops::Deref;

use cubecl_core::CubeDim;
use cubecl_matmul::components::{
    MatmulIdent, MatmulLineSizes, MatmulSetupError, MatrixLayout, TilingScheme,
    global::{
        GlobalConfig, PlaneRoleConfig, SpecializedLoadingSides, load::LoaderMode,
        multi_stage::EventLoadingMode,
    },
    stage::StageConfig,
};

use super::*;

/// Convolution specific config, extends regular matmul [`Config`](global::Config)
pub trait ConvGemmConfig: GlobalConfig {
    /// The size of the convolution kernel at `dim`
    fn kernel_size(&self, dim: u32) -> u32;
    /// The dilation of the kernel at `dim`
    fn dilation(&self, dim: u32) -> u32;
    /// The stride of the kernel at `dim`
    fn stride(&self, dim: u32) -> u32;
    /// The padding of the kernel at `dim`
    fn padding(&self, dim: u32) -> i32;
    /// The dimensionality of the kernel
    fn dimensionality(&self) -> Dimensionality;

    fn line_sizes(&self) -> MatmulLineSizes;
    fn check_spatial_bounds(&self) -> bool;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ConvolutionConfig<M: GlobalConfig> {
    matmul: M,
    pub kernel_size: [u32; 3],
    pub stride: [u32; 3],
    pub dilation: [u32; 3],
    pub padding: [i32; 3],
    dimensionality: Dimensionality,
    num_stages: u32,
}

impl<M: GlobalConfig> Deref for ConvolutionConfig<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.matmul
    }
}

impl<M: GlobalConfig> GlobalConfig for ConvolutionConfig<M> {
    type StageConfig = M::StageConfig;
    type StageMemoryConfig = <M::StageConfig as StageConfig>::StageMemoryConfig;

    fn stage_memory_config(&self) -> Self::StageMemoryConfig {
        self.stage_config().stage_memory_config()
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

    fn loader_mode(&self) -> LoaderMode {
        self.matmul.loader_mode()
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
    fn kernel_size(&self, dim: u32) -> u32 {
        self.kernel_size[dim as usize]
    }

    fn dilation(&self, dim: u32) -> u32 {
        self.dilation[dim as usize]
    }

    fn stride(&self, dim: u32) -> u32 {
        self.stride[dim as usize]
    }

    fn padding(&self, dim: u32) -> i32 {
        self.padding[dim as usize]
    }

    fn dimensionality(&self) -> Dimensionality {
        self.dimensionality
    }

    fn line_sizes(&self) -> cubecl_matmul::components::MatmulLineSizes {
        MatmulLineSizes {
            lhs: self.global_line_size(MatmulIdent::Lhs) as u8,
            rhs: self.global_line_size(MatmulIdent::Rhs) as u8,
            out: self.global_line_size(MatmulIdent::Out) as u8,
        }
    }

    fn check_spatial_bounds(&self) -> bool {
        let spatial_dims = self.dimensionality.num_dims();
        let mut has_padding = false;
        for i in 0..spatial_dims {
            has_padding |= self.padding[i as usize] != 0;
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

        let mut this = Self {
            matmul,
            kernel_size: [0; 3],
            stride: [0; 3],
            dilation: [0; 3],
            padding: [0; 3],
            dimensionality: dim,
            num_stages,
        };
        this.kernel_size[0..dims].copy_from_slice(kernel_size);
        this.stride[0..dims].copy_from_slice(stride);
        this.dilation[0..dims].copy_from_slice(dilation);
        this.padding[0..dims].copy_from_slice(padding);
        Ok(this)
    }

    pub fn to_matmul_config(self) -> M {
        self.matmul
    }
}
