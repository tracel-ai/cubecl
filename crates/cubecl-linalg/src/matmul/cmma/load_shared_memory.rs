use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{Dimensions, Offsets, SharedMemories},
    config::CmmaConfig,
};

use crate::matmul::cmma::block_io::{
    base::BlockLoader, horizontal_block_check::HorizontalCheckBlockIO,
    unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
    whole_block_check::WholeCheckBlockIO,
};

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    offsets: Offsets,
    mut shared: SharedMemories<FC>,
    dims: Dimensions,
    config: Comptime<CmmaConfig>,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let num_tiles_in_k = Comptime::runtime(block_size_k / tile_size);

    load_lhs(lhs, offsets, &mut shared.lhs, num_tiles_in_k, dims, config);
    load_rhs(rhs, offsets, &mut shared.rhs, num_tiles_in_k, dims, config);
}

#[cube]
pub(crate) fn load_lhs<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    offsets: Offsets,
    shared_lhs: &mut SharedMemory<FC>,
    num_tiles_in_k: UInt,
    dims: Dimensions,
    config: Comptime<CmmaConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);
    let tile_row = UNIT_POS_Y / num_tiles_in_k;
    let tile_col = UNIT_POS_Y % num_tiles_in_k;

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_k_bounds) {
            load_tile::<F, FC, WholeCheckBlockIO>(
                lhs,
                shared_lhs,
                offsets.batch_lhs,
                tile_row,
                tile_col,
                dims.m,
                dims.k,
                offsets.cube_row,
                offsets.k,
                config,
            );
        } else {
            load_tile::<F, FC, VerticalCheckBlockIO>(
                lhs,
                shared_lhs,
                offsets.batch_lhs,
                tile_row,
                tile_col,
                dims.m,
                dims.k,
                offsets.cube_row,
                offsets.k,
                config,
            );
        }
    } else if Comptime::get(check_k_bounds) {
        load_tile::<F, FC, HorizontalCheckBlockIO>(
            lhs,
            shared_lhs,
            offsets.batch_lhs,
            tile_row,
            tile_col,
            dims.m,
            dims.k,
            offsets.cube_row,
            offsets.k,
            config,
        );
    } else {
        load_tile::<F, FC, UncheckedBlockIO>(
            lhs,
            shared_lhs,
            offsets.batch_lhs,
            tile_row,
            tile_col,
            dims.m,
            dims.k,
            offsets.cube_row,
            offsets.k,
            config,
        );
    }
}

#[cube]
pub(crate) fn load_rhs<F: Float, FC: Float>(
    rhs: &Tensor<F>,
    offsets: Offsets,
    shared_rhs: &mut SharedMemory<FC>,
    k_tiles: UInt,
    dims: Dimensions,
    config: Comptime<CmmaConfig>,
) {
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);
    let tile_row = UNIT_POS_Y % k_tiles;
    let tile_col = UNIT_POS_Y / k_tiles;

    if Comptime::get(check_k_bounds) {
        if Comptime::get(check_n_bounds) {
            load_tile::<F, FC, WholeCheckBlockIO>(
                rhs,
                shared_rhs,
                offsets.batch_rhs,
                tile_row,
                tile_col,
                dims.k,
                dims.n,
                offsets.k,
                offsets.cube_col,
                config,
            );
        } else {
            load_tile::<F, FC, VerticalCheckBlockIO>(
                rhs,
                shared_rhs,
                offsets.batch_rhs,
                tile_row,
                tile_col,
                dims.k,
                dims.n,
                offsets.k,
                offsets.cube_col,
                config,
            );
        }
    } else if Comptime::get(check_n_bounds) {
        load_tile::<F, FC, HorizontalCheckBlockIO>(
            rhs,
            shared_rhs,
            offsets.batch_rhs,
            tile_row,
            tile_col,
            dims.k,
            dims.n,
            offsets.k,
            offsets.cube_col,
            config,
        );
    } else {
        load_tile::<F, FC, UncheckedBlockIO>(
            rhs,
            shared_rhs,
            offsets.batch_rhs,
            tile_row,
            tile_col,
            dims.k,
            dims.n,
            offsets.k,
            offsets.cube_col,
            config,
        );
    }
}
#[cube]
fn load_tile<F: Float, FC: Float, L: BlockLoader<F, FC>>(
    tensor: &Tensor<F>,
    shared_memory: &mut SharedMemory<FC>,
    batch_offset: UInt,
    tile_row: UInt,
    tile_col: UInt,
    dim_vertical: UInt,
    dim_horizontal: UInt,
    skip_row: UInt,
    skip_col: UInt,
    config: Comptime<CmmaConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let tile_size_r = Comptime::runtime(tile_size);
    let tensor_vec = Comptime::vectorization(tensor);
    let tensor_vec_r = Comptime::runtime(tensor_vec);

    // Must equal SUBCUBE_DIM, but must be known comptime too
    let coop_dim = Comptime::map(config, |c| c.coop_dim);

    let coop_id = UNIT_POS_Y;
    let lane_id = UNIT_POS_X;

    let num_unit_reads = tile_size * tile_size / (tensor_vec * coop_dim);
    let num_units_per_row = Comptime::runtime(tile_size / tensor_vec);

    let lane_row_step = Comptime::runtime(coop_dim * tensor_vec / tile_size);
    let lane_row_offset = lane_id / num_units_per_row;
    let read_row_offset = skip_row + tile_row * tile_size_r + lane_row_offset;

    let lane_col_offset = lane_id % num_units_per_row * tensor_vec_r;
    let read_col = skip_col + tile_col * tile_size_r + lane_col_offset;

    let sm_stride = Comptime::runtime(tile_size * tile_size);

    let write_offset = coop_id * sm_stride + lane_id * tensor_vec_r;
    let sm_step = Comptime::runtime(coop_dim * tensor_vec);
    // let sm_step = Comptime::runtime(tile_size * tensor_vec); // ?

    for i in range(0u32, Comptime::get(num_unit_reads), Comptime::new(true)) {
        let read_row = read_row_offset + i * lane_row_step;
        let write_pos = write_offset + i * sm_step;

        L::load_tile(
            tensor,
            shared_memory,
            batch_offset,
            read_row,
            read_col,
            write_pos,
            dim_vertical,
            dim_horizontal,
        )
    }
}
