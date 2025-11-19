use crate::components::{stage::matmul::partition::SharedPartitionMatmulConfig, tile::TileConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the plane partitioned stage matmul
pub struct PlanePartitionedStageConfig<TC: TileConfig> {
    pub shared: SharedPartitionMatmulConfig<TC>,
}

impl<TC: TileConfig> PlanePartitionedStageConfig<TC> {
    pub fn from_shared_partition_config(shared: SharedPartitionMatmulConfig<TC>) -> Self {
        Self { shared }
    }
}
