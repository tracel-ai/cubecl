use crate::components::{
    AttentionSetupError, AttentionTilingScheme,
    stage::{SharedPartitionAttentionConfig, StageAttentionConfig},
    tile::TileAttentionConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitPartitionStageConfig<TC: TileAttentionConfig> {
    pub shared: SharedPartitionAttentionConfig<TC>,
}

// impl<FC: TileAttentionConfig> StageAttentionConfig for UnitPartitionStageConfig<FC> {
//     type TileAttentionConfig = FC;

//     fn elements_in_partition_seq_q(&self) -> u32 {
//         todo!()
//     }

//     fn elements_in_partition_seq_kv(&self) -> u32 {
//         todo!()
//     }

//     fn elements_in_stage_seq_q(&self) -> u32 {
//         todo!()
//     }

//     fn elements_in_stage_seq_kv(&self) -> u32 {
//         todo!()
//     }

//     fn tile_config(&self) -> Self::TileAttentionConfig {
//         todo!()
//     }

//     fn elements_in_tile_seq_q(&self) -> u32 {
//         todo!()
//     }

//     fn elements_in_tile_seq_kv(&self) -> u32 {
//         todo!()
//     }

//     fn num_planes(&self) -> u32 {
//         todo!()
//     }

//     // fn plane_dim(&self) -> u32 {
//     //     self.tile_config.plane_dim()
//     // }

//     // fn num_planes(&self) -> u32 {
//     //     self.num_planes
//     // }

//     // fn tile_config(&self) -> Self::TileAttentionConfig {
//     //     self.tile_config
//     // }

//     // fn tiling_scheme(&self) -> AttentionTilingScheme {
//     //     self.tiling_scheme
//     // }

//     // fn reuse_key_value(&self) -> bool {
//     //     self.reuse_key_value
//     // }

//     // fn num_rows_per_unit(&self) -> u32 {
//     //     self.tile_config.num_rows_per_unit()
//     // }
// }
