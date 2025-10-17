use crate::components::attention_types::*;
use crate::components::tile::MaskTile;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::global::memory::{GlobalIterator, ViewDirection};
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::AttentionPrecision;
use crate::components::stage::StageAttentionConfig;
use cubecl_std::CubeOption;

#[derive(CubeType)]
pub struct LogicalIterator {
    row: u32,
    col: RuntimeCell<u32>,
    step_col: u32,
}

#[cube]
impl LogicalIterator {
    fn init(q_offset: u32, step_col: u32) -> LogicalIterator {
        LogicalIterator {
            row: q_offset,
            col: RuntimeCell::new(0),
            step_col,
        }
    }

    fn read(&self) -> Coords2d {
        (self.row, self.col.read())
    }

    fn advance(&mut self) {
        self.col.store(self.col.read() + self.step_col);
    }
}

#[derive(CubeType)]
pub enum MaskReader<AP: AttentionPrecision> {
    Materialized(GlobalIterator<Line<MSK<AP>>>, LogicalIterator),
    Logical(LogicalIterator),
}

#[cube]
impl<AP: AttentionPrecision> MaskReader<AP> {
    pub fn new_logical(q_offset: u32, step: u32) -> Self {
        MaskReader::<AP>::new_Logical(LogicalIterator::init(q_offset, step))
    }

    pub fn new_materialized(q_offset: u32, mask: View<Line<MSK<AP>>, Coords2d>, step: u32) -> Self {
        let mask = mask.slice((q_offset, 0), mask.shape());
        let global_iter = GlobalIterator::new(mask, step, ViewDirection::Col, false);

        MaskReader::<AP>::new_Materialized(global_iter, LogicalIterator::init(q_offset, step))
    }

    // TODO read tile too
    pub fn read<S: StageAttentionConfig>(
        &self,
        #[comptime] pos_in_partition: Coords2d,
        #[comptime] config: S,
    ) -> (Coords2d, CubeOption<StridedTile<MSK<AP>>>) {
        match self {
            MaskReader::Materialized(global_iterator, logical_iterator) => (
                logical_iterator.read(),
                CubeOption::new_Some(get_tile::<AP, S>(global_iterator, pos_in_partition, config)),
            ),
            MaskReader::Logical(logical_iterator) => {
                (logical_iterator.read(), CubeOption::new_None())
            }
        }
    }

    pub fn advance_view(&mut self) {
        match self {
            MaskReader::Logical(logical_iter) => logical_iter.advance(),
            MaskReader::Materialized(global_iter, logical_iter) => {
                global_iter.advance();
                logical_iter.advance()
            }
        }
    }
}

#[cube]
pub fn get_tile<AP: AttentionPrecision, S: StageAttentionConfig>(
    global_iter: &GlobalIterator<Line<MSK<AP>>>,
    #[comptime] tile: Coords2d,
    #[comptime] config: S,
) -> StridedTile<MSK<AP>> {
    let (row_in_partition, col) = tile;
    let attention_tile_size = config.tiling_scheme().tile_size;

    let row = row_in_partition + UNIT_POS_Y * config.tiling_scheme().partition_size.seq_q;

    let tile = StridedTile::<MSK<AP>>::new_strided(
        global_iter
            .view()
            .slice(
                (
                    row * attention_tile_size.seq_q,
                    col.runtime() * attention_tile_size.seq_kv,
                ),
                (attention_tile_size.seq_q, attention_tile_size.seq_kv).runtime(),
            )
            .to_linear_slice(),
        config.tiling_scheme().elements_in_partition_seq_kv(),
        MatrixLayout::RowMajor,
    );

    tile
}
