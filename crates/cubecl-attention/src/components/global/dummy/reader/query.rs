use crate::components::attention_types::*;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::AttentionPrecision;
use crate::components::stage::StageAttentionConfig;

#[derive(CubeType)]
pub struct QueryReader<AP: AttentionPrecision> {
    query: View<Line<QG<AP>>, Coords2d>,
}

#[cube]
impl<AP: AttentionPrecision> QueryReader<AP> {
    pub fn new(q_offset: u32, query: View<Line<QG<AP>>, Coords2d>) -> Self {
        let query = query.slice((q_offset, 0), query.shape());

        QueryReader::<AP> { query }
    }

    pub fn get_tile<S: StageAttentionConfig>(
        &self,
        tile: Coords2d,
        #[comptime] config: S,
    ) -> StridedTile<QG<AP>> {
        let (row_in_partition, col) = tile;
        let attention_tile_size = config.tiling_scheme().tile_size;

        let row = row_in_partition + UNIT_POS_Y * config.tiling_scheme().partition_size.seq_q;

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
            config.tiling_scheme().elements_in_partition_head_dim(),
            MatrixLayout::RowMajor,
        )
    }
}
