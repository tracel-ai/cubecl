use crate::components::{AttentionTileSize, attention_types::*};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::GlobalMemoryConfig;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::Swizzle;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::AttentionPrecision;
use crate::components::stage::AttentionPartitioner;

#[derive(CubeType)]
pub struct QueryReader<AP: AttentionPrecision> {
    query: View<Line<QG<AP>>, Coords2d>,
    #[cube(comptime)]
    gmem_config: GlobalMemoryConfig,
}

#[cube]
impl<AP: AttentionPrecision> QueryReader<AP> {
    pub fn new(
        stage_q_offset: u32,
        query: View<Line<QG<AP>>, Coords2d>,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let query = query.slice((stage_q_offset, 0), query.shape());

        QueryReader::<AP> { query, gmem_config }
    }

    pub fn get_tile<P: AttentionPartitioner>(
        &self,
        tile: Coords2d,
        #[comptime] attention_tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] partition_head_dim: u32,
    ) -> StridedTile<QG<AP>> {
        let (row_in_partition, col) = tile;

        let row = row_in_partition + P::seq_q_index() * partition_seq_q;

        let line_size = self.gmem_config.line_size;

        let slice = self
            .query
            .slice(
                (
                    row * attention_tile_size.seq_q,
                    col * attention_tile_size.head_dim,
                ),
                (attention_tile_size.seq_q, attention_tile_size.head_dim).runtime(),
            )
            .to_linear_slice();

        let start = 0;
        let length = attention_tile_size.seq_q * attention_tile_size.head_dim / line_size;
        let end = start + length;
        let stride = partition_head_dim * attention_tile_size.head_dim / line_size;

        StridedTile::<QG<AP>>::new_strided(
            slice,
            start,
            end,
            stride,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
            line_size,
        )
    }
}
