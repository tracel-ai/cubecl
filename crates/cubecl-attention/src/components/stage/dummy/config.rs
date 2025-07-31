use cubecl_matmul::components::{
    GlobalPartitionSize, MatrixLayout, StageIdent, TilingScheme, stage::StageMemoryConfig,
    tile::TileConfig,
};

use crate::components::{AttentionSetupError, stage::StageAttentionConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct DummyStageConfig<ST: TileConfig, VT: TileConfig> {
    score_stage_memory_config: AttentionStageMemoryConfig<ST>,
    value_stage_memory_config: AttentionStageMemoryConfig<VT>,
    num_planes: u32,
}

impl<ST: TileConfig, VT: TileConfig> StageAttentionConfig for DummyStageConfig<ST, VT> {
    type ScoreConfig = ST;
    type ScoreStageMemoryConfig = AttentionStageMemoryConfig<ST>;

    type ValueConfig = VT;
    type ValueStageMemoryConfig = AttentionStageMemoryConfig<VT>;

    fn plane_dim(&self) -> u32 {
        self.score_config().plane_dim()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn rows_per_plane(&self) -> u32 {
        // self.tiling_scheme...
        8
    }

    fn score_stage_memory_config(&self) -> Self::ScoreStageMemoryConfig {
        self.score_stage_memory_config
    }

    fn value_stage_memory_config(&self) -> Self::ValueStageMemoryConfig {
        self.value_stage_memory_config
    }

    fn score_config(&self) -> Self::ScoreConfig {
        self.score_stage_memory_config.tile_config()
    }

    fn value_config(&self) -> Self::ValueConfig {
        self.value_stage_memory_config.tile_config()
    }
}

impl<ST: TileConfig, VT: TileConfig> DummyStageConfig<ST, VT> {
    pub fn new(
        score_stage_memory_config: AttentionStageMemoryConfig<ST>,
        value_stage_memory_config: AttentionStageMemoryConfig<VT>,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            score_stage_memory_config,
            value_stage_memory_config,
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
        self.tile_config
    }

    fn num_main_flow_planes(&self) -> u32 {
        todo!()
    }

    fn tiling_scheme(&self) -> TilingScheme {
        TilingScheme {
            tile_size: (8, 8, 8).into(),
            partition_size: (1, 1, 1).into(),
            stage_size: (1, 1, 1).into(),
            global_partition_size: GlobalPartitionSize::new(1, 1, 1),
        }
    }

    fn stage_line_size(&self, ident: StageIdent) -> u32 {
        1
    }

    fn matrix_layout(&self, ident: StageIdent) -> MatrixLayout {
        todo!()
    }

    fn num_stages(&self, ident: StageIdent) -> u32 {
        todo!()
    }
}

impl<T: TileConfig> AttentionStageMemoryConfig<T> {
    pub fn new(tile_config: T) -> Self {
        Self { tile_config }
    }
}
