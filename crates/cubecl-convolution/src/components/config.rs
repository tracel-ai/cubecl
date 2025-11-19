use std::ops::Deref;

use cubecl_core::CubeDim;
use cubecl_matmul::components::{
    MatmulIdent, MatmulLineSizes, MatmulSetupError, MatrixLayout, TilingScheme,
    global::RoleRuleConfig,
    global::{
        GlobalConfig, PlaneRoleConfig, SpecializedLoadingSides, multi_stage::EventLoadingMode,
        read::ReaderMode,
    },
    stage::{StageConfig, StageMemoryConfig, SwizzleMode, TilingLayoutEnum},
};

use super::*;

// /// Convolution specific config, extends regular matmul [`Config`](global::Config)
// pub trait ConvGemmConfig: GlobalConfig {
//     /// The size of the convolution kernel at `dim`
//     fn convolution_params(&self) -> ConvolutionParams;
//     fn line_sizes(&self) -> MatmulLineSizes;
//     fn check_spatial_bounds(&self) -> bool;
// }

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

// impl<M: GlobalConfig> GlobalConfig for ConvolutionConfig<M> {
//     type StageConfig = M::StageConfig;
//     type LhsReaderConfig = M::LhsReaderConfig;
//     type RhsReaderConfig = M::RhsReaderConfig;

//     fn lhs_reader_config(&self) -> Self::LhsReaderConfig {
//         self.matmul.lhs_reader_config()
//     }

//     fn rhs_reader_config(&self) -> Self::RhsReaderConfig {
//         self.matmul.rhs_reader_config()
//     }

//     fn stage_config(&self) -> Self::StageConfig {
//         self.matmul.stage_config()
//     }

//     fn global_line_size(&self, ident: MatmulIdent) -> u32 {
//         self.matmul.global_line_size(ident)
//     }

//     fn matrix_layout(&self, ident: MatmulIdent) -> MatrixLayout {
//         self.matmul.matrix_layout(ident)
//     }

//     fn swizzle_mode(&self, ident: MatmulIdent) -> SwizzleMode {
//         self.matmul.swizzle_mode(ident)
//     }

//     fn tiling_layout(&self, ident: MatmulIdent) -> TilingLayoutEnum {
//         self.matmul.tiling_layout(ident)
//     }

//     fn num_loading_planes(&self, ident: MatmulIdent) -> u32 {
//         self.matmul.num_loading_planes(ident)
//     }

//     fn plane_dim(&self) -> u32 {
//         self.matmul.plane_dim()
//     }

//     fn check_row_bounds(&self, ident: MatmulIdent) -> bool {
//         self.matmul.check_row_bounds(ident)
//     }

//     fn check_col_bounds(&self, ident: MatmulIdent) -> bool {
//         self.matmul.check_col_bounds(ident)
//     }

//     fn check_k_bounds(&self) -> bool {
//         self.matmul.check_k_bounds()
//     }

//     fn num_stages(&self, _ident: MatmulIdent) -> u32 {
//         self.num_stages
//     }

//     fn tiling_scheme(&self) -> TilingScheme {
//         self.matmul.tiling_scheme()
//     }

//     fn cube_dim(&self) -> CubeDim {
//         CubeDim::new(
//             <Self as GlobalConfig>::plane_dim(self),
//             <Self as GlobalConfig>::tiling_scheme(self).tiles_in_stage_m(),
//             1,
//         )
//     }

//     fn role_rule_config(&self) -> RoleRuleConfig {
//         self.matmul.role_rule_config()
//     }
// }

impl<M: GlobalConfig> ConvolutionConfig<M> {
    fn line_sizes(&self) -> cubecl_matmul::components::MatmulLineSizes {
        self.matmul.global_line_sizes()
    }

    fn check_spatial_bounds(&self) -> bool {
        let spatial_dims = self.convolution_params.dimensionality.num_dims();
        let mut has_padding = false;
        for i in 0..spatial_dims {
            has_padding |= self.convolution_params.padding[i as usize] != 0;
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
            convolution_params: params,
            num_stages,
        })
    }

    // pub fn to_matmul_config(self) -> M {
    //     self.matmul
    // }
}
