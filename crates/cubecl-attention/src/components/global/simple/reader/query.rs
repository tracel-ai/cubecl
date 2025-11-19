use crate::components::{
    AttentionTileSize, attention_types::*, global::simple::attention, tile::TileAttentionConfig,
};
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
        #[comptime] config: S,
    ) -> StridedTile<QG<AP>> {
        let (row_in_partition, col) = tile;

        // get from smem_config
        // config.tile_config().attention_tile_size();
        let attention_tile_size = comptime!(AttentionTileSize {
            seq_q: todo!(),
            head_dim: todo!(),
            seq_kv: todo!(),
            val_dim: todo!()
        });
        let partition_seq_q: u32 = todo!(); // config.partition_size.seq_q;
        let elements_in_partition_head_dim: u32 = todo!(); //             config.tiling_scheme().elements_in_partition_head_dim(),

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
            elements_in_partition_head_dim,
            Swizzle::none(),
            MatrixLayout::RowMajor,
            // TODO
            999u32,
        );
        todo!()
    }
}
