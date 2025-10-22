use cubecl_core::CubeDim;
use cubecl_matmul::components::{
    MatrixLayout, StageIdent, global::memory::GlobalMemoryConfig, stage::StageMemoryConfig,
};

use crate::components::{
    AttentionIdent, AttentionSetupError, AttentionTilingScheme, global::GlobalAttentionConfig,
    stage::StageAttentionConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct SimpleGlobalConfig<S: StageAttentionConfig> {
    stage_config: S,
    num_planes: u32,
    causal_mask: bool,
}

impl<S: StageAttentionConfig> GlobalAttentionConfig for SimpleGlobalConfig<S> {
    type StageConfig = S;

    fn score_stage_memory_config(&self) -> StageMemoryConfig {
        self.stage_config
            .score_stage_memory_config()
            .into_matmul_config(StageIdent::Rhs)
    }

    fn value_stage_memory_config(&self) -> StageMemoryConfig {
        self.stage_config
            .value_stage_memory_config()
            .into_matmul_config(StageIdent::Rhs)
    }

    fn stage_config(&self) -> S {
        self.stage_config
    }

    fn cube_dim(&self) -> CubeDim {
        CubeDim::new_2d(self.plane_dim(), self.num_planes)
    }

    fn plane_dim(&self) -> u32 {
        self.stage_config.plane_dim()
    }

    fn global_memory_config(&self, ident: AttentionIdent) -> GlobalMemoryConfig {
        let tiling_scheme = self.stage_config.tiling_scheme();

        let elements_in_tile_row = tiling_scheme.tile_size.num_rows(ident);
        let elements_in_tile_col = tiling_scheme.tile_size.num_cols(ident);
        let elements_in_stage_row =
            tiling_scheme.partition_size.num_rows(ident) * elements_in_tile_row;
        let elements_in_stage_col =
            tiling_scheme.partition_size.num_cols(ident) * elements_in_tile_col;

        GlobalMemoryConfig {
            elements_in_tile_row,
            elements_in_tile_col,
            elements_in_stage_row,
            elements_in_stage_col,
            global_line_size: 1,
            check_row_bounds: false,
            check_col_bounds: false,
            matrix_layout: MatrixLayout::RowMajor,
        }
    }

    fn tiling_scheme(&self) -> AttentionTilingScheme {
        self.stage_config.tiling_scheme()
    }

    fn causal_mask(&self) -> bool {
        self.causal_mask
    }
}

impl<S: StageAttentionConfig> SimpleGlobalConfig<S> {
    pub fn new(
        stage_config: S,
        num_planes: u32,
        causal_mask: bool,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            stage_config,
            num_planes,
            causal_mask,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}
