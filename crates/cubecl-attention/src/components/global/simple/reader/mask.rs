use crate::components::attention_types::*;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatrixLayout;
use cubecl_matmul::components::global::memory::{GlobalIterator, ViewDirection};
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::layout::Coordinates;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::AttentionPrecision;
use crate::components::stage::{AttentionPartitioner, StageAttentionConfig};
use cubecl_std::CubeOption;

#[derive(CubeType)]
pub struct LogicalIterator {
    row: u32,
    col: RuntimeCell<u32>,
    step_col: u32,
}

#[cube]
impl LogicalIterator {
    fn init(stage_q_offset: u32, step_col: u32) -> LogicalIterator {
        LogicalIterator {
            row: stage_q_offset,
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
pub struct MaterializedMaskReader<M: Numeric> {
    global_iter: GlobalIterator<Line<M>>,
    logical_iter: LogicalIterator,
    // TODO not sure if mandatory, but i need for the stride when reading in global memory
    seq_kv_shape: u32,
}

#[derive(CubeType)]
pub enum MaskReader<AP: AttentionPrecision> {
    Materialized(MaterializedMaskReader<MSK<AP>>),
    Logical(LogicalIterator),
}

#[cube]
impl<AP: AttentionPrecision> MaskReader<AP> {
    pub fn new_logical(partition_q_offset: u32, step: u32) -> Self {
        MaskReader::<AP>::new_Logical(LogicalIterator::init(partition_q_offset, step))
    }

    pub fn new_materialized(
        stage_q_offset: u32,
        partition_q_offset: u32,
        mask: View<Line<MSK<AP>>, Coords2d>,
        step: u32,
        seq_kv_shape: u32,
    ) -> Self {
        let mask = mask.slice((stage_q_offset, 0), mask.shape());
        let global_iter = GlobalIterator::new(mask, step, ViewDirection::Col, false);

        MaskReader::<AP>::new_Materialized(MaterializedMaskReader::new(
            global_iter,
            LogicalIterator::init(partition_q_offset, step),
            seq_kv_shape,
        ))
    }

    pub fn read<P: AttentionPartitioner, S: StageAttentionConfig>(
        &self,
        #[comptime] pos_in_partition: Coords2d,
        #[comptime] config: S,
    ) -> (Coords2d, CubeOption<StridedTile<MSK<AP>>>) {
        match self {
            MaskReader::Materialized(materialized_mask_reader) => {
                materialized_mask_reader.read::<P, S>(pos_in_partition, config)
            }
            MaskReader::Logical(logical_iter) => (
                Coords2d::add(logical_iter.read(), pos_in_partition.runtime()),
                CubeOption::new_None(),
            ),
        }
    }

    pub fn advance_view(&mut self) {
        match self {
            MaskReader::Logical(logical_iter) => logical_iter.advance(),
            MaskReader::Materialized(materialized_mask_reader) => {
                materialized_mask_reader.advance()
            }
        }
    }
}

#[cube]
impl<M: Numeric> MaterializedMaskReader<M> {
    fn new(
        global_iter: GlobalIterator<Line<M>>,
        logical_iter: LogicalIterator,
        seq_kv_shape: u32,
    ) -> Self {
        MaterializedMaskReader::<M> {
            global_iter,
            logical_iter,
            seq_kv_shape,
        }
    }

    fn read<P: AttentionPartitioner, S: StageAttentionConfig>(
        &self,
        #[comptime] pos_in_partition: Coords2d,
        #[comptime] config: S,
    ) -> (Coords2d, CubeOption<StridedTile<M>>) {
        let (row_in_partition, col) = pos_in_partition;
        let attention_tile_size = config.tiling_scheme().tile_size;

        let row = row_in_partition + P::seq_q_index() * config.tiling_scheme().partition_size.seq_q;

        let tile = StridedTile::<M>::new_strided(
            self.global_iter
                .view()
                .slice(
                    (
                        row * attention_tile_size.seq_q,
                        col.runtime() * attention_tile_size.seq_kv,
                    ),
                    (attention_tile_size.seq_q, attention_tile_size.seq_kv).runtime(),
                )
                .to_linear_slice(),
            self.seq_kv_shape,
            MatrixLayout::RowMajor,
        );

        (
            Coords2d::add(self.logical_iter.read(), pos_in_partition.runtime()),
            CubeOption::new_Some(tile),
        )
    }

    fn advance(&mut self) {
        self.global_iter.advance();
        self.logical_iter.advance()
    }
}
