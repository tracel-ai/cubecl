use cubecl_matmul::components::{
    MatrixLayout, StageIdent, TilingScheme, stage::StageMemoryConfig, tile::TileConfig,
};

use crate::components::{AttentionSetupError, stage::StageAttentionConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<ST: TileConfig, VT: TileConfig> {
    score_config: ST,
    value_config: VT,
    num_planes: u32,
}

impl<ST: TileConfig, VT: TileConfig> StageAttentionConfig for DummyStageConfig<ST, VT> {
    type ScoreConfig = ST;
    type ScoreStageMemoryConfig = AttentionStageMemoryConfig<ST>;

    type ValueConfig = VT;
    type ValueStageMemoryConfig = AttentionStageMemoryConfig<VT>;

    fn plane_dim(&self) -> u32 {
        self.score_config.plane_dim()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn rows_per_plane(&self) -> u32 {
        // self.tiling_scheme...
        8
    }
}

impl<ST: TileConfig, VT: TileConfig> DummyStageConfig<ST, VT> {
    pub fn new(
        score_config: ST,
        value_config: VT,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            score_config,
            value_config,
            num_planes,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        Ok(self)
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionStageMemoryConfig<T: TileConfig> {
    tile_config: T,
}

impl<T: TileConfig> StageMemoryConfig for AttentionStageMemoryConfig<T> {
    type TileConfig = T;

    fn tile_config(self) -> Self::TileConfig {
        todo!()
    }

    fn num_main_flow_planes(&self) -> u32 {
        todo!()
    }

    fn tiling_scheme(&self) -> TilingScheme {
        todo!()
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        todo!()
    }

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        todo!()
    }

    fn num_stages(&self, ident: StageIdent) -> u32 {
        todo!()
    }
}
