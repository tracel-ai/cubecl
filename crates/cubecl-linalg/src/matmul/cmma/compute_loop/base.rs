use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::super::prologue::{Fragments, Ids, SharedMemories};
use crate::matmul::cmma::config::{SmemLoaderStrategy, TilingOrderStrategy};
use crate::matmul::cmma::{
    config::ComptimeCmmaInfo,
    load_shared_memory::{
        base::SmemLoader,
        continuous::ContinuousSmemLoader,
        load_info::{LhsLoadInfo, LoadInfo, RhsLoadInfo},
        tiled_layout::{ColMajorTiling, RowMajorTiling, TilingOrder},
        tilewise::TilewiseSmemLoader,
    },
};

#[cube]
pub(crate) trait ComputeLoop {
    fn compute_loop<F: Float, FC: Float>(
        shared_memories: SharedMemories<FC>,
        fragments: &mut Fragments<F, FC>,
        ids: Ids,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
}

#[cube]
pub(crate) fn load_tile_into_fragment<FC: Float, I: LoadInfo>(
    nth_tile: u32,
    smem: SharedMemory<FC>,
    fragment: &cmma::Matrix<FC>,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let smem_stride = I::num_tile_elements(comptime_info);

    let smem_pos = nth_tile * smem_stride;
    let slice = smem.slice(smem_pos, smem_pos + smem_stride);
    cmma::load::<FC>(fragment, slice, I::tile_width(comptime_info));
}

#[cube]
pub(crate) fn get_smem_position_lhs<F: Float, FC: Float>(
    tile_row: u32,
    tile_col: u32,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> u32 {
    match comptime_info.lhs_smem_loader_strategy {
        SmemLoaderStrategy::Tilewise(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                get_tile_smem_position::<F, FC, LhsLoadInfo, RowMajorTiling, TilewiseSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
            TilingOrderStrategy::ColMajor => {
                get_tile_smem_position::<F, FC, LhsLoadInfo, ColMajorTiling, TilewiseSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
        },
        SmemLoaderStrategy::Continuous(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                get_tile_smem_position::<F, FC, LhsLoadInfo, RowMajorTiling, ContinuousSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
            TilingOrderStrategy::ColMajor => {
                get_tile_smem_position::<F, FC, LhsLoadInfo, ColMajorTiling, ContinuousSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
        },
    }
}

#[cube]
pub(crate) fn get_smem_position_rhs<F: Float, FC: Float>(
    tile_row: u32,
    tile_col: u32,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> u32 {
    match comptime_info.rhs_smem_loader_strategy {
        SmemLoaderStrategy::Tilewise(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                get_tile_smem_position::<F, FC, RhsLoadInfo, RowMajorTiling, TilewiseSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
            TilingOrderStrategy::ColMajor => {
                get_tile_smem_position::<F, FC, RhsLoadInfo, ColMajorTiling, TilewiseSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
        },
        SmemLoaderStrategy::Continuous(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                get_tile_smem_position::<F, FC, RhsLoadInfo, RowMajorTiling, ContinuousSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
            TilingOrderStrategy::ColMajor => {
                get_tile_smem_position::<F, FC, RhsLoadInfo, ColMajorTiling, ContinuousSmemLoader>(
                    tile_row,
                    tile_col,
                    comptime_info,
                )
            }
        },
    }
}

#[cube]
fn get_tile_smem_position<
    F: Float,
    FC: Float,
    I: LoadInfo,
    T: TilingOrder,
    S: SmemLoader<F, FC, I, T>,
>(
    tile_row: u32,
    tile_col: u32,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> u32 {
    S::get_tile_smem_index(tile_row, tile_col, comptime_info)
}
