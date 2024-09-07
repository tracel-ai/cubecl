use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Accumulators, Dimensions, Offsets},
    block_io::{
        base::BlockWriter, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
    config::CmmaConfig,
};

#[cube]
pub(crate) fn write_to_output<F: Float>(
    out: &mut Tensor<F>,
    accumulators: Accumulators<F>,
    offsets: Offsets,
    dims: Dimensions,
    #[comptime] config: CmmaConfig,
) {
    let accumulator_sm = fragment_to_shared_memory(accumulators);
    shared_memory_to_output(out, offsets, accumulator_sm, dims, config);
}

#[cube]
fn fragment_to_shared_memory<F: Float>(accumulators: Accumulators<F>) -> SharedMemory<F> {
    let mut acc_sm = SharedMemory::<F>::new(4096);

    let coop_id = UNIT_POS_Y;
    let slice_offset_0 = coop_id * 512;
    let slice_offset_1 = slice_offset_0 + 256;
    let slice_offset_2 = slice_offset_1 + 256;

    let slice = acc_sm.slice_mut(slice_offset_0, slice_offset_1);
    cmma::store(slice, &accumulators.first, 16, cmma::MatrixLayout::RowMajor);

    let slice = acc_sm.slice_mut(slice_offset_1, slice_offset_2);
    cmma::store(
        slice,
        &accumulators.second,
        16,
        cmma::MatrixLayout::RowMajor,
    );

    acc_sm
}

#[cube]
pub(crate) fn shared_memory_to_output<F: Float>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    #[comptime] config: CmmaConfig,
) {
    let check_m_bounds = config.check_m_bounds;
    let check_n_bounds = config.check_n_bounds;

    if check_m_bounds {
        if check_n_bounds {
            write_tile::<F, WholeCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
        } else {
            write_tile::<F, VerticalCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
        }
    } else if check_n_bounds {
        write_tile::<F, HorizontalCheckBlockIO>(out, offsets, accumulator_sm, dims, config);
    } else {
        write_tile::<F, UncheckedBlockIO>(out, offsets, accumulator_sm, dims, config);
    }
}

#[cube]
fn write_tile<F: Float, W: BlockWriter<F>>(
    out: &mut Tensor<F>,
    offsets: Offsets,
    accumulator_sm: SharedMemory<F>,
    dims: Dimensions,
    #[comptime] config: CmmaConfig,
) {
    // Other values not supported
    let n_tiles = 2;

    let tile_size = config.tile_size;
    let out_vec = out.vectorization_factor();
    let n_units_per_tile_row = tile_size / out_vec;
    let num_tile_elems = tile_size * tile_size;

    let coop_dim = 32;
    let coop_id = UNIT_POS_Y;
    let lane_id = UNIT_POS_X;

    let tile_row = coop_id / n_tiles;
    let tile_col = (coop_id % n_tiles) * n_tiles;

    let read_offset = n_tiles * coop_id * num_tile_elems;
    let read_0 = read_offset + lane_id * out_vec;
    let read_1 = read_0 + coop_dim * out_vec;

    let unit_write_row_0 = lane_id / n_units_per_tile_row;
    let unit_write_row_1 = unit_write_row_0 + coop_dim / out_vec;
    let unit_write_col = (lane_id % n_units_per_tile_row) * n_units_per_tile_row;

    let row_offset = offsets.cube_row + tile_row * tile_size;
    let write_row_0 = row_offset + unit_write_row_0;
    let write_row_1 = row_offset + unit_write_row_1;
    let write_col = offsets.cube_col + tile_col * tile_size + unit_write_col;

    W::write_output(
        out,
        accumulator_sm,
        0,
        offsets.batch_out,
        read_0,
        write_row_0,
        write_col,
        dims,
        config,
    );
    W::write_output(
        out,
        accumulator_sm,
        0,
        offsets.batch_out,
        read_1,
        write_row_1,
        write_col,
        dims,
        config,
    );
    W::write_output(
        out,
        accumulator_sm,
        1,
        offsets.batch_out,
        read_0,
        write_row_0,
        write_col,
        dims,
        config,
    );
    W::write_output(
        out,
        accumulator_sm,
        1,
        offsets.batch_out,
        read_1,
        write_row_1,
        write_col,
        dims,
        config,
    );
}
