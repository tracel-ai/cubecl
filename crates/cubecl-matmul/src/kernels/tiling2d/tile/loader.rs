use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, CubeType};
use std::marker::PhantomData;

use crate::kernels::tiling2d::{
    config::CubeTiling2dConfig,
    load_shared_memory::{LoadInfo, Loader},
};

use super::{
    block_io::base::BlockLoader,
    memory_access::{MatchingVectorization, UnmatchingVectorization},
};

// Transposed tensor's vectorization must be 1
// Plain tensor's vectorization must equal tile size
pub(crate) struct TileLoader<N: Numeric> {
    _f: PhantomData<N>,
}

#[derive(CubeType)]
pub(crate) struct LoadIndices {
    pub offset: u32,
    pub gm_stride: u32,
    pub sm_stride: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct CheckBounds {
    pub dim_vertical: u32,
    pub dim_horizontal: u32,
    pub skip_row: u32,
    pub skip_col: u32,
}

#[derive(CubeType, Copy, Clone)]
pub(crate) struct ReadTileInfo {
    pub read_row: u32,
    pub read_col: u32,
    pub gm_position_base: u32,
    pub sm_position_base: u32,
    pub gm_stride: u32,
    pub sm_stride: u32,
}

#[cube]
impl<N: Numeric> Loader<N> for TileLoader<N> {
    fn load_lhs_plain<B: BlockLoader<N>>(
        lhs: &Tensor<Line<N>>,
        load_info: LoadInfo<N>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = lhs.stride(lhs.rank() - 1);

        let load_indices = LoadIndices {
            offset: coordinates.skip_row + load_info.k * gm_stride + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_n,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.k,
            dim_horizontal: dims.m,
            skip_row: load_info.k,
            skip_col: coordinates.skip_row,
        };

        load_plain::<N, B>(lhs, load_info, load_indices, check_bounds, config);
    }

    fn load_lhs_transposed<B: BlockLoader<N>>(
        lhs: &Tensor<Line<N>>,
        load_info: LoadInfo<N>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = lhs.stride(lhs.rank() - 2);

        let load_indices = LoadIndices {
            offset: coordinates.skip_row * gm_stride + load_info.k + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_m,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.m,
            dim_horizontal: dims.k,
            skip_row: coordinates.skip_row,
            skip_col: load_info.k,
        };

        load_transposed::<N, B>(lhs, load_info, load_indices, check_bounds, config);
    }

    fn load_rhs_plain<B: BlockLoader<N>>(
        rhs: &Tensor<Line<N>>,
        load_info: LoadInfo<N>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let coordinates = load_info.coordinates;
        let dims = load_info.dims;
        let gm_stride = rhs.stride(rhs.rank() - 2);

        let load_indices = LoadIndices {
            offset: coordinates.skip_col + load_info.k * gm_stride + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_n,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.k,
            dim_horizontal: dims.n,
            skip_row: load_info.k,
            skip_col: coordinates.skip_col,
        };

        load_plain::<N, B>(rhs, load_info, load_indices, check_bounds, config);
    }

    fn load_rhs_transposed<B: BlockLoader<N>>(
        rhs: &Tensor<Line<N>>,
        load_info: LoadInfo<N>,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let dims = load_info.dims;
        let coordinates = load_info.coordinates;
        let gm_stride = rhs.stride(rhs.rank() - 1);

        let load_indices = LoadIndices {
            offset: coordinates.skip_col * gm_stride + load_info.k + load_info.batch_offset,
            gm_stride,
            sm_stride: config.block_size_n,
        };
        let check_bounds = CheckBounds {
            dim_vertical: dims.n,
            dim_horizontal: dims.k,
            skip_row: coordinates.skip_col,
            skip_col: load_info.k,
        };

        load_transposed::<N, B>(rhs, load_info, load_indices, check_bounds, config);
    }
}

#[cube]
pub(crate) fn load_plain<N: Numeric, L: BlockLoader<N>>(
    tensor: &Tensor<Line<N>>,
    load_info: LoadInfo<N>,
    load_indices: LoadIndices,
    check_bounds: CheckBounds,
    #[comptime] config: CubeTiling2dConfig,
) {
    let coordinates = load_info.coordinates;

    let line_size = tensor.line_size();
    let tile_size = config.tile_size;
    let sm_dim_vertical = config.block_size_k;

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_row;
    let write_col = coordinates.unit_col;

    let gm_position_base = read_row * load_indices.gm_stride + read_col + load_indices.offset;
    let sm_position_base = write_row * load_indices.sm_stride + write_col;

    let read_tile_info = ReadTileInfo {
        read_row,
        read_col,
        gm_position_base,
        sm_position_base,
        gm_stride: load_indices.gm_stride,
        sm_stride: load_indices.sm_stride,
    };
    let mut sm = load_info.shared_memory;

    if write_row < sm_dim_vertical {
        if comptime![line_size == tile_size] {
            L::load_tile_plain::<MatchingVectorization>(
                tensor,
                &mut sm,
                read_tile_info,
                config,
                check_bounds,
            );
        } else {
            L::load_tile_plain::<UnmatchingVectorization>(
                tensor,
                &mut sm,
                read_tile_info,
                config,
                check_bounds,
            );
        }
    }
}

#[cube]
pub(crate) fn load_transposed<N: Numeric, L: BlockLoader<N>>(
    tensor: &Tensor<Line<N>>,
    load_info: LoadInfo<N>,
    load_indices: LoadIndices,
    check_bounds: CheckBounds,
    #[comptime] config: CubeTiling2dConfig,
) {
    let coordinates = load_info.coordinates;

    let sm_dim_vertical = config.block_size_k;

    let read_row = coordinates.unit_row;
    let read_col = coordinates.unit_col;
    let write_row = coordinates.unit_col;
    let write_col = coordinates.unit_row;

    let gm_position_base = read_row * load_indices.gm_stride + read_col + load_indices.offset;
    let sm_position_base = write_row * load_indices.sm_stride + write_col;

    let read_tile_info = ReadTileInfo {
        read_row,
        read_col,
        gm_position_base,
        sm_position_base,
        gm_stride: load_indices.gm_stride,
        sm_stride: load_indices.sm_stride,
    };
    let mut sm = load_info.shared_memory;

    if write_row < sm_dim_vertical {
        L::load_tile_transposed(tensor, &mut sm, read_tile_info, config, check_bounds);
    }
}
