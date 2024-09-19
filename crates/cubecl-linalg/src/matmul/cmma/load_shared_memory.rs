use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{RuntimeCmmaInfo, SharedMemories},
    config::ComptimeCmmaInfo,
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
    k_offset: u32,
    mut shared: SharedMemories<FC>,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let block_size_k = comptime_info.block_size_k;
    let tile_size = comptime_info.tile_size;
    let num_tiles_in_k = block_size_k / tile_size;

    load_lhs(
        lhs,
        &mut shared.lhs,
        num_tiles_in_k,
        k_offset,
        runtime_info,
        comptime_info,
    );
    load_rhs(
        rhs,
        &mut shared.rhs,
        num_tiles_in_k,
        k_offset,
        runtime_info,
        comptime_info,
    );
}

#[cube]
pub(crate) fn load_lhs<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    shared_lhs: &mut SharedMemory<FC>,
    num_tiles_in_k: u32,
    k_offset: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let check_m_bounds = comptime_info.check_m_bounds;
    let check_k_bounds = comptime_info.check_k_bounds;
    let ids = runtime_info.ids;
    let dims = runtime_info.dims;
    let offsets = runtime_info.offsets;

    let tile_row = ids.coop / num_tiles_in_k;
    let tile_col = ids.coop % num_tiles_in_k;

    if check_m_bounds {
        if check_k_bounds {
            load_tile::<F, FC, WholeCheckBlockIO>(
                lhs,
                shared_lhs,
                offsets.batch_lhs,
                tile_row,
                tile_col,
                dims.m,
                dims.k,
                offsets.cube_row,
                k_offset,
                runtime_info,
                comptime_info,
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
                k_offset,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_k_bounds {
        load_tile::<F, FC, HorizontalCheckBlockIO>(
            lhs,
            shared_lhs,
            offsets.batch_lhs,
            tile_row,
            tile_col,
            dims.m,
            dims.k,
            offsets.cube_row,
            k_offset,
            runtime_info,
            comptime_info,
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
            k_offset,
            runtime_info,
            comptime_info,
        );
    }
}

#[cube]
pub(crate) fn load_rhs<F: Float, FC: Float>(
    rhs: &Tensor<F>,
    shared_rhs: &mut SharedMemory<FC>,
    num_tiles_in_k: u32,
    k_offset: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let check_k_bounds = comptime_info.check_k_bounds;
    let check_n_bounds = comptime_info.check_n_bounds;
    let ids = runtime_info.ids;
    let dims = runtime_info.dims;
    let offsets = runtime_info.offsets;

    let tile_row = ids.coop % num_tiles_in_k;
    let tile_col = ids.coop / num_tiles_in_k;

    if check_k_bounds {
        if check_n_bounds {
            load_tile::<F, FC, WholeCheckBlockIO>(
                rhs,
                shared_rhs,
                offsets.batch_rhs,
                tile_row,
                tile_col,
                dims.k,
                dims.n,
                k_offset,
                offsets.cube_col,
                runtime_info,
                comptime_info,
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
                k_offset,
                offsets.cube_col,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_n_bounds {
        load_tile::<F, FC, HorizontalCheckBlockIO>(
            rhs,
            shared_rhs,
            offsets.batch_rhs,
            tile_row,
            tile_col,
            dims.k,
            dims.n,
            k_offset,
            offsets.cube_col,
            runtime_info,
            comptime_info,
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
            k_offset,
            offsets.cube_col,
            runtime_info,
            comptime_info,
        );
    }
}
#[cube]
fn load_tile<F: Float, FC: Float, L: BlockLoader<F, FC>>(
    tensor: &Tensor<F>,
    shared_memory: &mut SharedMemory<FC>,
    batch_offset: u32,
    tile_row: u32,
    tile_col: u32,
    dim_vertical: u32,
    dim_horizontal: u32,
    skip_row: u32,
    skip_col: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let tile_size = comptime_info.tile_size;
    let tensor_vec = vectorization_of(tensor);
    let ids = runtime_info.ids;

    // Must equal SUBCUBE_DIM, but must be known comptime too
    let coop_dim = comptime_info.coop_dim;

    let num_unit_reads = tile_size * tile_size / (tensor_vec * coop_dim);
    let num_units_per_row = tile_size / tensor_vec;

    let lane_row_step = coop_dim * tensor_vec / tile_size;
    let lane_row_offset = ids.lane / num_units_per_row;
    let read_row_offset = skip_row + tile_row * tile_size + lane_row_offset;

    let lane_col_offset = ids.lane % num_units_per_row * tensor_vec;
    let read_col = skip_col + tile_col * tile_size + lane_col_offset;

    let sm_stride = tile_size * tile_size;

    let write_offset = ids.coop * sm_stride + ids.lane * tensor_vec;
    let sm_step = coop_dim * tensor_vec;

    #[unroll]
    for i in 0..num_unit_reads {
        let read_row = read_row_offset + i * lane_row_step;
        let write_pos = write_offset + i * sm_step;

        L::load_single(
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
