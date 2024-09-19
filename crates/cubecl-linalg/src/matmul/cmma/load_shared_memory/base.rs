use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

use crate::matmul::cmma::base::SharedMemories;
use crate::matmul::cmma::block_io::base::BlockLoader;
use crate::matmul::cmma::block_io::{
    horizontal_block_check::HorizontalCheckBlockIO, unchecked_block::UncheckedBlockIO,
    vertical_block_check::VerticalCheckBlockIO, whole_block_check::WholeCheckBlockIO,
};
use crate::matmul::cmma::load_shared_memory::{
    continous::ContinuousSmemLoader,
    load_info::{LhsLoadInfo, RhsLoadInfo},
    tiled_layout::ColMajorTiling,
    tiled_layout::RowMajorTiling,
    tilewise::TilewiseSmemLoader,
};
use crate::matmul::cmma::{base::RuntimeCmmaInfo, config::ComptimeCmmaInfo};

use super::load_info::LoadInfo;

#[cube]
pub(crate) trait SmemLoader<F: Float, FC: Float> {
    fn load_gmem_to_smem<I: LoadInfo, L: BlockLoader<F, FC>>(
        gmem: &Tensor<F>,
        smem: &mut SharedMemory<FC>,
        k_offset: u32,
        runtime_info: RuntimeCmmaInfo,
        #[comptime] comptime_info: ComptimeCmmaInfo,
    );
}

#[cube]
pub(crate) fn load_lhs<F: Float, FC: Float, S: SmemLoader<F, FC>>(
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
            S::load_gmem_to_smem::<LhsLoadInfo, WholeCheckBlockIO>(
                lhs,
                shared_lhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        } else {
            S::load_gmem_to_smem::<LhsLoadInfo, VerticalCheckBlockIO>(
                lhs,
                shared_lhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_k_bounds {
        S::load_gmem_to_smem::<LhsLoadInfo, HorizontalCheckBlockIO>(
            lhs,
            shared_lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else {
        S::load_gmem_to_smem::<LhsLoadInfo, UncheckedBlockIO>(
            lhs,
            shared_lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    }
}

#[cube]
pub(crate) fn load_rhs<F: Float, FC: Float, S: SmemLoader<F, FC>>(
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
            S::load_gmem_to_smem::<RhsLoadInfo, WholeCheckBlockIO>(
                rhs,
                shared_rhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        } else {
            S::load_gmem_to_smem::<RhsLoadInfo, VerticalCheckBlockIO>(
                rhs,
                shared_rhs,
                k_offset,
                runtime_info,
                comptime_info,
            );
        }
    } else if check_k_bounds {
        S::load_gmem_to_smem::<RhsLoadInfo, HorizontalCheckBlockIO>(
            rhs,
            shared_rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else {
        S::load_gmem_to_smem::<RhsLoadInfo, UncheckedBlockIO>(
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
    if comptime_info.lhs_smem_loader_strategy == 0 {
        load_lhs::<F, FC, TilewiseSmemLoader<RowMajorTiling>>(
            lhs,
            &mut shared.lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else if comptime_info.lhs_smem_loader_strategy == 1 {
        load_lhs::<F, FC, TilewiseSmemLoader<ColMajorTiling>>(
            lhs,
            &mut shared.lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else if comptime_info.lhs_smem_loader_strategy == 2 {
        load_lhs::<F, FC, ContinuousSmemLoader<RowMajorTiling>>(
            lhs,
            &mut shared.lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else {
        load_lhs::<F, FC, ContinuousSmemLoader<ColMajorTiling>>(
            lhs,
            &mut shared.lhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    }

    if comptime_info.rhs_smem_loader_strategy == 0 {
        load_rhs::<F, FC, TilewiseSmemLoader<RowMajorTiling>>(
            rhs,
            &mut shared.rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else if comptime_info.rhs_smem_loader_strategy == 1 {
        load_rhs::<F, FC, TilewiseSmemLoader<ColMajorTiling>>(
            rhs,
            &mut shared.rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else if comptime_info.rhs_smem_loader_strategy == 2 {
        load_rhs::<F, FC, ContinuousSmemLoader<RowMajorTiling>>(
            rhs,
            &mut shared.rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    } else {
        load_rhs::<F, FC, ContinuousSmemLoader<ColMajorTiling>>(
            rhs,
            &mut shared.rhs,
            k_offset,
            runtime_info,
            comptime_info,
        );
    }
}
