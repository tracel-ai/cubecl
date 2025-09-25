use crate::components::{global::memory::GlobalMemoryConfig, tile::io::TileKind};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<EO: Numeric>: CubeType + 'static + Send + Sync {
    /// Tile kind beigng stored
    type TileKind: TileKind<ReadWrite>;

    /// Writes the given slice to global memory, at a position that depends on
    /// plane and accumulator indexes.
    fn write<ES: Numeric>(
        this: &mut Self,
        smem_tile: &<Self::TileKind as TileKind<ReadWrite>>::Tile<ES>,
        tile: Coords2d,
        #[comptime] plane_dim: u32,
        #[comptime] config: GlobalMemoryConfig,
    );
}
