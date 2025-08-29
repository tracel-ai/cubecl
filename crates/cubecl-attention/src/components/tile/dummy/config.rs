use cubecl_matmul::components::{
    GlobalPartitionSize, MatrixLayout, StageIdent, TilingScheme, stage::StageMemoryConfig,
    tile::TileConfig,
};

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
            tile_size: *self.tile_config.tile_size(),
            partition_size: (1, 1, 1).into(),
            stage_size: (1, 1, 1).into(),
            global_partition_size: GlobalPartitionSize::new(1, 1, 1),
        }
    }

    fn stage_line_size(&self, _ident: StageIdent) -> u32 {
        1
    }

    fn matrix_layout(&self, _ident: StageIdent) -> MatrixLayout {
        MatrixLayout::RowMajor
    }

    fn num_stages(&self, _ident: StageIdent) -> u32 {
        1
    }
}

impl<T: TileConfig> AttentionStageMemoryConfig<T> {
    pub fn new(tile_config: T) -> Self {
        Self { tile_config }
    }
}
