use crate::components::{stage::matmul::partition::SharedPartitionMatmulConfig, tile::TileConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the unit partitioned stage matmul
pub struct UnitPartitionedStageConfig<TC: TileConfig> {
    pub shared: SharedPartitionMatmulConfig<TC>,
}

impl<TC: TileConfig> UnitPartitionedStageConfig<TC> {
    pub fn from_shared_partition_config(shared: SharedPartitionMatmulConfig<TC>) -> Self {
        Self { shared }
    }
}
