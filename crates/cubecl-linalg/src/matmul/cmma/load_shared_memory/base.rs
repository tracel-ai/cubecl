use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use super::super::prologue::{RuntimeCmmaInfo, SharedMemories};
use crate::matmul::cmma::block_io::base::BlockLoader;
use crate::matmul::cmma::block_io::{
    horizontal_block_check::HorizontalCheckBlockIO, unchecked_block::UncheckedBlockIO,
    vertical_block_check::VerticalCheckBlockIO, whole_block_check::WholeCheckBlockIO,
};
use crate::matmul::cmma::config::ComptimeCmmaInfo;

use super::load_info::LoadInfo;
use super::tiled_layout::TilingOrder;
use super::{
    continuous::ContinuousSmemLoader,
    load_info::{LhsLoadInfo, RhsLoadInfo},
    tiled_layout::{ColMajorTiling, RowMajorTiling},
    tilewise::TilewiseSmemLoader,
};
use crate::matmul::cmma::config::{SmemLoaderStrategy, TilingOrderStrategy};

#[cube]
pub(crate) trait SmemLoader<F: Float, FC: Float, I: LoadInfo, T: TilingOrder> {
    fn load_gmem_to_smem<L: BlockLoader<F, FC>>(
        gmem: &Tensor<F>,
        smem: &mut SharedMemory<FC>,
        k_offset: u32,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );

    fn get_tile_smem_index(
        tile_row: u32,
        tile_col: u32,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    ) -> u32;
}

#[cube]
pub(crate) fn get_tile_smem_index<I: LoadInfo, T: TilingOrder>(
    tile_row: u32,
    tile_col: u32,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) -> u32 {
    let smem_tile_width = I::smem_tile_width(comptime_info);
    let smem_tile_height = I::smem_tile_height(comptime_info);

    T::to_nth_tile(tile_row, tile_col, smem_tile_width, smem_tile_height)
}

#[cube]
pub(crate) fn load_lhs<
    F: Float,
    FC: Float,
    I: LoadInfo,
    T: TilingOrder,
    S: SmemLoader<F, FC, I, T>,
>(
    lhs: &Tensor<F>,
    shared_lhs: &mut SharedMemory<FC>,
    k_offset: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let check_m_bounds = comptime_info.check_m_bounds;
    let check_k_bounds = comptime_info.check_k_bounds;

    if check_m_bounds {
        if check_k_bounds {
            S::load_gmem_to_smem::<WholeCheckBlockIO>(
                lhs,
                shared_lhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        } else {
            S::load_gmem_to_smem::<VerticalCheckBlockIO>(
                lhs,
                shared_lhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_k_bounds {
        S::load_gmem_to_smem::<HorizontalCheckBlockIO>(
            lhs,
            shared_lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else {
        S::load_gmem_to_smem::<UncheckedBlockIO>(
            lhs,
            shared_lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    }
}

#[cube]
pub(crate) fn load_rhs<
    F: Float,
    FC: Float,
    I: LoadInfo,
    T: TilingOrder,
    S: SmemLoader<F, FC, I, T>,
>(
    rhs: &Tensor<F>,
    shared_rhs: &mut SharedMemory<FC>,
    k_offset: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    let check_m_bounds = comptime_info.check_m_bounds;
    let check_k_bounds = comptime_info.check_k_bounds;

    if check_m_bounds {
        if check_k_bounds {
            S::load_gmem_to_smem::<WholeCheckBlockIO>(
                rhs,
                shared_rhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        } else {
            S::load_gmem_to_smem::<VerticalCheckBlockIO>(
                rhs,
                shared_rhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_k_bounds {
        S::load_gmem_to_smem::<HorizontalCheckBlockIO>(
            rhs,
            shared_rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else {
        S::load_gmem_to_smem::<UncheckedBlockIO>(
            rhs,
            shared_rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    }
}

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, FC: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    mut shared: SharedMemories<FC>,
    k_offset: u32,
    runtime_info: RuntimeCmmaInfo,
    #[comptime] comptime_info: ComptimeCmmaInfo,
) {
    match comptime_info.lhs_smem_loader_strategy {
        SmemLoaderStrategy::Tilewise(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                load_lhs::<F, FC, LhsLoadInfo, RowMajorTiling, TilewiseSmemLoader>(
                    lhs,
                    &mut shared.lhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                );
            }
            TilingOrderStrategy::ColMajor => {
                load_lhs::<F, FC, LhsLoadInfo, ColMajorTiling, TilewiseSmemLoader>(
                    lhs,
                    &mut shared.lhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                )
            }
        },
        SmemLoaderStrategy::Continuous(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                load_lhs::<F, FC, LhsLoadInfo, RowMajorTiling, ContinuousSmemLoader>(
                    lhs,
                    &mut shared.lhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                )
            }
            TilingOrderStrategy::ColMajor => {
                load_lhs::<F, FC, LhsLoadInfo, ColMajorTiling, ContinuousSmemLoader>(
                    lhs,
                    &mut shared.lhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                )
            }
        },
    }

    match comptime_info.rhs_smem_loader_strategy {
        SmemLoaderStrategy::Tilewise(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                load_rhs::<F, FC, RhsLoadInfo, RowMajorTiling, TilewiseSmemLoader>(
                    rhs,
                    &mut shared.rhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                );
            }
            TilingOrderStrategy::ColMajor => {
                load_rhs::<F, FC, RhsLoadInfo, ColMajorTiling, TilewiseSmemLoader>(
                    rhs,
                    &mut shared.rhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                )
            }
        },
        SmemLoaderStrategy::Continuous(tiling_order) => match tiling_order {
            TilingOrderStrategy::RowMajor => {
                load_rhs::<F, FC, RhsLoadInfo, RowMajorTiling, ContinuousSmemLoader>(
                    rhs,
                    &mut shared.rhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                )
            }
            TilingOrderStrategy::ColMajor => {
                load_rhs::<F, FC, RhsLoadInfo, ColMajorTiling, ContinuousSmemLoader>(
                    rhs,
                    &mut shared.rhs,
                    k_offset,
                    runtime_info,
                    comptime_info,
                )
            }
        },
    }
}
