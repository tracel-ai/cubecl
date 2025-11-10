use cubecl_core::CubeDim;
use cubecl_matmul::components::{
    MatrixLayout, global::memory::GlobalMemoryConfig, stage::StageMemoryConfig,
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

    fn key_stage_memory_config(&self) -> StageMemoryConfig {
        let tiling_scheme = self.stage_config.tiling_scheme();

        StageMemoryConfig {
            num_main_flow_planes: 1,
            elements_in_tile_row: tiling_scheme.elements_in_tile_seq_kv(),
            elements_in_tile_col: tiling_scheme.elements_in_tile_head_dim(),
            tiles_in_stage_row: tiling_scheme.tiles_in_stage_seq_kv(),
            tiles_in_stage_col: tiling_scheme.tiles_in_stage_head_dim(),
            stage_line_size: 1,
            matrix_layout: MatrixLayout::RowMajor,
            num_stages: 1,
        }
    }

    fn value_stage_memory_config(&self) -> StageMemoryConfig {
        let tiling_scheme = self.stage_config.tiling_scheme();

        StageMemoryConfig {
            num_main_flow_planes: 1,
            elements_in_tile_row: tiling_scheme.elements_in_tile_seq_kv(),
            elements_in_tile_col: tiling_scheme.elements_in_tile_val_dim(),
            tiles_in_stage_row: tiling_scheme.tiles_in_stage_seq_kv(),
            tiles_in_stage_col: tiling_scheme.tiles_in_stage_val_dim(),
            stage_line_size: 1,
            matrix_layout: MatrixLayout::RowMajor,
            num_stages: 1,
        }
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

        GlobalMemoryConfig::new(
            elements_in_tile_row,
            elements_in_tile_col,
            elements_in_stage_row,
            elements_in_stage_col,
            1,
            false,
            false,
            MatrixLayout::RowMajor,
        )
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
