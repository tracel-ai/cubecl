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
    let k_tiles = Comptime::runtime(block_size_k / tile_size);

    load_lhs(lhs, offsets, &mut shared.lhs, k_tiles, dims, config);
    load_rhs(rhs, offsets, &mut shared.rhs, k_tiles, dims, config);
}

#[cube]
pub(crate) fn load_lhs<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    offsets: Offsets,
    shared_lhs: &mut SharedMemory<FC>,
    k_tiles: UInt,
    dims: Dimensions,
    config: Comptime<CmmaConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_k_bounds) {
            load_tile::<F, FC, WholeCheckBlockIO>(
                lhs,
                shared_lhs,
                offsets.batch_lhs,
                UNIT_POS_Y / k_tiles,
                UNIT_POS_Y % k_tiles,
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
                UNIT_POS_Y / k_tiles,
                UNIT_POS_Y % k_tiles,
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
            UNIT_POS_Y / k_tiles,
            UNIT_POS_Y % k_tiles,
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
            UNIT_POS_Y / k_tiles,
            UNIT_POS_Y % k_tiles,
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

    if Comptime::get(check_k_bounds) {
        if Comptime::get(check_n_bounds) {
            load_tile::<F, FC, WholeCheckBlockIO>(
                rhs,
                shared_rhs,
                offsets.batch_rhs,
                UNIT_POS_Y % k_tiles,
                UNIT_POS_Y / k_tiles,
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
                UNIT_POS_Y % k_tiles,
                UNIT_POS_Y / k_tiles,
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
            UNIT_POS_Y % k_tiles,
            UNIT_POS_Y / k_tiles,
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
            UNIT_POS_Y % k_tiles,
            UNIT_POS_Y / k_tiles,
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

    // Will likely fail if SUBCUBE_DIM is not 32
    let coop_dim = UInt::new(32);
    let coop_id = UNIT_POS_Y;
    let lane_id = UNIT_POS_X;

    // There are two rows because 16x16 tiles with 32 threads -> 2 vec4 loads
    let unit_read_row_0 = lane_id / tensor_vec_r;
    let unit_read_row_1 = unit_read_row_0 + coop_dim / tensor_vec_r;
    let read_row_0 = skip_row + tile_row * tile_size_r + unit_read_row_0;
    let read_row_1 = skip_row + tile_row * tile_size_r + unit_read_row_1;

    let unit_read_col = lane_id % tensor_vec_r * tensor_vec_r;
    let read_col = skip_col + tile_col * tile_size_r + unit_read_col;

    let sm_stride = Comptime::runtime(tile_size * tile_size);
    let write_pos_0 = coop_id * sm_stride + lane_id * tensor_vec_r;
    let write_pos_1 = write_pos_0 + sm_stride / UInt::new(2);

    L::load_tile(
        tensor,
        shared_memory,
        batch_offset,
        read_row_0,
        read_col,
        write_pos_0,
        dim_vertical,
        dim_horizontal,
    );
    L::load_tile(
        tensor,
        shared_memory,
        batch_offset,
        read_row_1,
        read_col,
        write_pos_1,
        dim_vertical,
        dim_horizontal,
    );
}
