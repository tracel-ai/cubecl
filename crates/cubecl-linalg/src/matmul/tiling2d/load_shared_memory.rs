use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{
    base::{BatchOffsets, Coordinates, Dimensions, SharedMemories},
    config::CubeTiling2dConfig,
    tile::block_io::{
        base::BlockLoader, horizontal_block_check::HorizontalCheckBlockIO,
        unchecked_block::UncheckedBlockIO, vertical_block_check::VerticalCheckBlockIO,
        whole_block_check::WholeCheckBlockIO,
    },
};

#[derive(CubeType)]
#[allow(dead_code)]
pub(crate) struct LoadInfo<F: Float> {
    pub coordinates: Coordinates,
    pub k: UInt,
    pub batch_offset: UInt,
    pub shared_memory: SharedMemory<F>,
    pub config: Comptime<CubeTiling2dConfig>,
    pub dims: Dimensions,
}

#[cube]
pub(crate) trait Loader<F: Float>: Sync + Send + 'static {
    fn load_lhs_plain<B: BlockLoader<F>>(lhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_lhs_transposed<B: BlockLoader<F>>(lhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_rhs_plain<B: BlockLoader<F>>(rhs: &Tensor<F>, load_info: LoadInfo<F>);
    fn load_rhs_transposed<B: BlockLoader<F>>(rhs: &Tensor<F>, load_info: LoadInfo<F>);
}

#[cube]
pub(crate) fn load_to_shared_memories<F: Float, L: Loader<F>>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    coordinates: Coordinates,
    k: UInt,
    offsets: BatchOffsets,
    shared: SharedMemories<F>,
    config: Comptime<CubeTiling2dConfig>,
    dims: Dimensions,
) {
    let lhs_transposed = Comptime::map(config, |c| c.lhs_transposed);
    let rhs_transposed = Comptime::map(config, |c| c.rhs_transposed);

    let lhs_load_info = LoadInfo {
        coordinates,
        k,
        batch_offset: offsets.lhs,
        shared_memory: shared.lhs,
        config,
        dims,
    };
    let rhs_load_info = LoadInfo {
        coordinates,
        k,
        batch_offset: offsets.rhs,
        shared_memory: shared.rhs,
        config,
        dims,
    };

    // Lhs must be loaded as transposed. If it already is transposed in global memory, we load as plain.
    if Comptime::get(lhs_transposed) {
        load_lhs_plain::<F, L>(lhs, lhs_load_info, config);
    } else {
        load_lhs_transposed::<F, L>(lhs, lhs_load_info, config);
    }

    // Rhs must be loaded as plain. If it is transposed in global memory, we transpose it back.
    if Comptime::get(rhs_transposed) {
        load_rhs_transposed::<F, L>(rhs, rhs_load_info, config);
    } else {
        load_rhs_plain::<F, L>(rhs, rhs_load_info, config);
    }
}

#[cube]
pub(crate) fn load_lhs_transposed<F: Float, L: Loader<F>>(
    lhs: &Tensor<F>,
    load_info: LoadInfo<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_k_bounds) {
            L::load_lhs_transposed::<WholeCheckBlockIO>(lhs, load_info);
        } else {
            L::load_lhs_transposed::<VerticalCheckBlockIO>(lhs, load_info);
        }
    } else if Comptime::get(check_k_bounds) {
        L::load_lhs_transposed::<HorizontalCheckBlockIO>(lhs, load_info);
    } else {
        L::load_lhs_transposed::<UncheckedBlockIO>(lhs, load_info);
    }
}

#[cube]
pub(crate) fn load_lhs_plain<F: Float, L: Loader<F>>(
    lhs: &Tensor<F>,
    load_info: LoadInfo<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    if Comptime::get(check_k_bounds) {
        if Comptime::get(check_m_bounds) {
            L::load_lhs_plain::<WholeCheckBlockIO>(lhs, load_info);
        } else {
            L::load_lhs_plain::<VerticalCheckBlockIO>(lhs, load_info);
        }
    } else if Comptime::get(check_m_bounds) {
        L::load_lhs_plain::<HorizontalCheckBlockIO>(lhs, load_info);
    } else {
        L::load_lhs_plain::<UncheckedBlockIO>(lhs, load_info);
    }
}

#[cube]
pub(crate) fn load_rhs_transposed<F: Float, L: Loader<F>>(
    rhs: &Tensor<F>,
    load_info: LoadInfo<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    if Comptime::get(check_n_bounds) {
        if Comptime::get(check_k_bounds) {
            L::load_rhs_transposed::<WholeCheckBlockIO>(rhs, load_info);
        } else {
            L::load_rhs_transposed::<VerticalCheckBlockIO>(rhs, load_info);
        }
    } else if Comptime::get(check_k_bounds) {
        L::load_rhs_transposed::<HorizontalCheckBlockIO>(rhs, load_info);
    } else {
        L::load_rhs_transposed::<UncheckedBlockIO>(rhs, load_info);
    }
}

#[cube]
pub(crate) fn load_rhs_plain<F: Float, L: Loader<F>>(
    rhs: &Tensor<F>,
    load_info: LoadInfo<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    if Comptime::get(check_k_bounds) {
        if Comptime::get(check_n_bounds) {
            L::load_rhs_plain::<WholeCheckBlockIO>(rhs, load_info);
        } else {
            L::load_rhs_plain::<VerticalCheckBlockIO>(rhs, load_info);
        }
    } else if Comptime::get(check_n_bounds) {
        L::load_rhs_plain::<HorizontalCheckBlockIO>(rhs, load_info);
    } else {
        L::load_rhs_plain::<UncheckedBlockIO>(rhs, load_info);
    }
}
