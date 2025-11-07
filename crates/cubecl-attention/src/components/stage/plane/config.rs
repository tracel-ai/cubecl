use crate::components::{
    AttentionSetupError, AttentionTilingScheme,
    stage::{AttentionStageMemoryConfig, StageAttentionConfig},
    tile::FragmentAttentionConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlanePartitionStageConfig<FC: FragmentAttentionConfig> {
    fragment_config: FC,
    score_stage_memory_config: AttentionStageMemoryConfig,
    value_stage_memory_config: AttentionStageMemoryConfig,
    tiling_scheme: AttentionTilingScheme,
    reuse_key_value: bool,
    num_planes: u32,
}

impl<FC: FragmentAttentionConfig> StageAttentionConfig for PlanePartitionStageConfig<FC> {
    type FragmentAttentionConfig = FC;

    fn plane_dim(&self) -> u32 {
        self.fragment_config.plane_dim()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn tile_config(&self) -> Self::FragmentAttentionConfig {
        self.fragment_config
    }

    fn score_stage_memory_config(&self) -> AttentionStageMemoryConfig {
        self.score_stage_memory_config
    }

    fn value_stage_memory_config(&self) -> AttentionStageMemoryConfig {
        self.value_stage_memory_config
    }

    fn tiling_scheme(&self) -> AttentionTilingScheme {
        self.tiling_scheme
    }

    fn reuse_key_value(&self) -> bool {
        self.reuse_key_value
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.fragment_config.num_rows_per_unit()
    }
}

impl<FC: FragmentAttentionConfig> PlanePartitionStageConfig<FC> {
    pub fn new(
        fragment_config: FC,
        score_stage_memory_config: AttentionStageMemoryConfig,
        value_stage_memory_config: AttentionStageMemoryConfig,
        tiling_scheme: AttentionTilingScheme,
        reuse_key_value: bool,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            fragment_config,
            score_stage_memory_config,
            value_stage_memory_config,
            tiling_scheme,
            reuse_key_value,
            num_planes,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, AttentionSetupError> {
        if self.reuse_key_value
            && (self.tiling_scheme.tile_size.head_dim != self.tiling_scheme.tile_size.val_dim
                || self.tiling_scheme.partition_size.head_dim
                    != self.tiling_scheme.partition_size.val_dim)
        {
            return Err(AttentionSetupError::InvalidConfig(Box::new(
        "When reusing key/value, head_dim must equal val_dim in both tile_size and partition_size."
            .to_string(),
    )));
        }

        Ok(self)
    }
}
