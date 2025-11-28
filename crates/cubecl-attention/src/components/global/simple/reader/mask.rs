use crate::components::tile::TileAttentionConfig;
use crate::components::{AttentionTileSize, attention_types::*};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::global::memory::{GlobalIterator, GlobalMemoryConfig};
use cubecl_matmul::components::tile::StridedTile;
use cubecl_std::tensor::{View, layout::Coords2d};
use cubecl_std::{Swizzle, tensor::layout::Coordinates};

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
    #[cube(comptime)]
    gmem_config: GlobalMemoryConfig,
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
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let mask = mask.slice((stage_q_offset, 0), mask.shape());
        let global_iter = GlobalIterator::new(mask, step, gmem_config.view_direction, false);

        MaskReader::<AP>::new_Materialized(MaterializedMaskReader::new(
            global_iter,
            LogicalIterator::init(partition_q_offset, step),
            seq_kv_shape,
            gmem_config,
        ))
    }

    pub fn read<P: AttentionPartitioner, S: StageAttentionConfig>(
        &self,
        #[comptime] pos_in_partition: Coords2d,
        #[comptime] config: S,
    ) -> (Coords2d, CubeOption<StridedTile<MSK<AP>>>) {
        let partition_tile_offset = (
            pos_in_partition.0 * config.elements_in_tile_seq_q(),
            pos_in_partition.1 * config.elements_in_tile_seq_kv(),
        );

        let (origin, tile) = match self {
            MaskReader::Materialized(materialized_mask_reader) => (
                materialized_mask_reader.logical_iter.read(),
                CubeOption::new_Some(materialized_mask_reader.read::<P>(
                    partition_tile_offset,
                    config.tile_config().attention_tile_size(),
                    config.elements_in_partition_seq_q(),
                )),
            ),
            MaskReader::Logical(logical_iter) => (logical_iter.read(), CubeOption::new_None()),
        };

        (Coords2d::add(origin, partition_tile_offset.runtime()), tile)
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
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        MaterializedMaskReader::<M> {
            global_iter,
            logical_iter,
            seq_kv_shape,
            gmem_config,
        }
    }

    fn read<P: AttentionPartitioner>(
        &self,
        #[comptime] partition_tile_offset: Coords2d,
        #[comptime] attention_tile_size: AttentionTileSize,
        #[comptime] elements_in_partition_seq_q: u32,
    ) -> StridedTile<M> {
        let (row_offset, col) = partition_tile_offset;

        let row = row_offset + P::seq_q_index() * elements_in_partition_seq_q;

        let slice = self
            .global_iter
            .view()
            .slice(
                (row, col.runtime()),
                (attention_tile_size.seq_q, attention_tile_size.seq_kv).runtime(),
            )
            .to_linear_slice();

        let line_size = self.gmem_config.line_size;
        let start = 0;
        let length = attention_tile_size.seq_q * attention_tile_size.seq_kv / line_size;
        let end = start + length;
        let stride = self.seq_kv_shape / line_size;

        StridedTile::<M>::new_strided(
            slice,
            start,
            end,
            stride,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
            line_size,
        )
    }

    fn advance(&mut self) {
        self.global_iter.advance();
        self.logical_iter.advance()
    }
}
