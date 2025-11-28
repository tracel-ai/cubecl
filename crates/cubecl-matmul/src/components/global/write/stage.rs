use crate::components::stage::RowMajorTilingOrder;
use crate::components::stage::StridedStageMemory;
use crate::components::stage::{ContiguousTilingLayout, StageFamily};
use crate::components::stage::{StageMemoryConfig, TilingLayout};
use crate::components::tile::StridedTile;
use crate::components::{stage::Stage, tile::io::Strided};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

pub type WriteTiling = ContiguousTilingLayout<RowMajorTilingOrder>;

pub struct PartitionedStageFamily;

impl StageFamily<ReadWrite> for PartitionedStageFamily {
    type TileKind = Strided;

    type Stage<ES: Numeric, T: TilingLayout> = PartitionedStage<ES>;
}

#[derive(CubeType, Clone, Copy)]
/// Layoutless stage for current writers. Tile only depends on the unit index, not the out tile.
pub struct PartitionedStage<ES: Numeric> {
    /// Underlying shared memory
    _smem: SharedMemory<Line<ES>>,
    pub unit_tile: StridedTile<ES, ReadWrite>,
}

#[cube]
impl<ES: Numeric> PartitionedStage<ES> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(unit_pos: Coords2d, #[comptime] config: StageMemoryConfig) -> PartitionedStage<ES> {
        let config = comptime![StageMemoryConfig {
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            ..config
        }];

        let inner = StridedStageMemory::<ES, WriteTiling>::new(config);

        let tile = inner.get_tile_mut(unit_pos);

        PartitionedStage::<ES> {
            _smem: inner.smem,
            unit_tile: tile,
        }
    }
}

#[cube]
impl<ES: Numeric> Stage<ES, ReadWrite> for PartitionedStage<ES> {
    type TileKind = Strided;

    fn tile(this: &Self, _tile: Coords2d) -> StridedTile<ES, ReadWrite> {
        this.unit_tile
    }
}
