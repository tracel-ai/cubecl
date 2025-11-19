use crate::components::{AttentionTileSize, attention_types::*};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::{
    Swizzle,
    tensor::{View, layout::Coords2d},
};

use crate::components::AttentionPrecision;
use crate::components::stage::{AttentionPartitioner, StageAttentionConfig};

#[derive(CubeType)]
pub struct QueryReader<AP: AttentionPrecision> {
    query: View<Line<QG<AP>>, Coords2d>,
}

#[cube]
impl<AP: AttentionPrecision> QueryReader<AP> {
    pub fn new(stage_q_offset: u32, query: View<Line<QG<AP>>, Coords2d>) -> Self {
        let query = query.slice((stage_q_offset, 0), query.shape());

        QueryReader::<AP> { query }
    }

    pub fn get_tile<P: AttentionPartitioner, S: StageAttentionConfig>(
        &self,
        tile: Coords2d,
        #[comptime] attention_tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] partition_head_dim: u32,
    ) -> StridedTile<QG<AP>> {
        comment!("query get_tile");

        let (row_in_partition, col) = tile;

        let row = row_in_partition + P::seq_q_index() * partition_seq_q;

        StridedTile::<QG<AP>>::new_strided(
            self.query
                .slice(
                    (
                        row * attention_tile_size.seq_q,
                        col * attention_tile_size.head_dim,
                    ),
                    (attention_tile_size.seq_q, attention_tile_size.head_dim).runtime(),
                )
                .to_linear_slice(),
            0,
            attention_tile_size.seq_q * attention_tile_size.head_dim,
            partition_head_dim * attention_tile_size.head_dim,
            Swizzle::none(),
            MatrixLayout::RowMajor,
            1u32,
        )
    }
}
