use crate::components::{
    AttentionSetupError, AttentionTilingScheme, stage::StageAttentionConfig,
    tile::TileAttentionConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlanePartitionStageConfig<TC: TileAttentionConfig> {
    tile_config: TC,
    tiling_scheme: AttentionTilingScheme,
    reuse_key_value: bool,
    num_planes: u32,
}

impl<TC: TileAttentionConfig> StageAttentionConfig for PlanePartitionStageConfig<TC> {
    type TileAttentionConfig = TC;

    fn plane_dim(&self) -> u32 {
        self.tile_config.plane_dim()
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn tile_config(&self) -> Self::TileAttentionConfig {
        self.tile_config
    }

    fn tiling_scheme(&self) -> AttentionTilingScheme {
        self.tiling_scheme
    }

    fn reuse_key_value(&self) -> bool {
        self.reuse_key_value
    }

    fn num_rows_per_unit(&self) -> u32 {
        self.tile_config.num_rows_per_unit()
    }
}

impl<TC: TileAttentionConfig> PlanePartitionStageConfig<TC> {
    pub fn new(
        tile_config: TC,
        tiling_scheme: AttentionTilingScheme,
        reuse_key_value: bool,
        num_planes: u32,
    ) -> Result<Self, AttentionSetupError> {
        Self {
            tile_config,
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
