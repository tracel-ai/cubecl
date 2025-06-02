use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, CubeType};

use crate::kernels::tiling2d::config::CubeTiling2dConfig;

use super::loader::{CheckBounds, ReadTileInfo};

#[derive(CubeType)]
pub(crate) struct WritePositions {
    pub out: u32,
    pub result: u32,
}

#[cube]
pub(crate) trait ContiguousAccess<N: Numeric>: Send + Sync + 'static {
    fn read_contiguous_unchecked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N>;

    fn read_contiguous_checked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        check_bounds: CheckBounds,
        read_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N>;

    fn write_contiguous_unchecked(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        positions: WritePositions,
        #[comptime] config: CubeTiling2dConfig,
    );

    fn write_contiguous_checked(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        positions: WritePositions,
        check_bounds: CheckBounds,
        write_col: u32,
        #[comptime] config: CubeTiling2dConfig,
    );
}

#[cube]
pub(crate) trait StridedAccess<N: Numeric>: Send + Sync + 'static {
    fn read_strided_unchecked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        gm_stride: u32,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N>;

    fn read_strided_checked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        gm_stride: u32,
        check_bounds: CheckBounds,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N>;
}

/// When vectorization == tile_size
pub(crate) struct MatchingVectorization;

/// When vectorization != tile_size
pub(crate) struct UnmatchingVectorization;

#[cube]
impl<N: Numeric> ContiguousAccess<N> for MatchingVectorization {
    fn read_contiguous_unchecked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        #[comptime] _config: CubeTiling2dConfig,
    ) -> Line<N> {
        tensor[gm_position]
    }

    fn read_contiguous_checked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        _check_bounds: CheckBounds,
        _read_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N> {
        // If vectorization matches, then it's certain to fit since tile_size divides block_sizes
        MatchingVectorization::read_contiguous_unchecked(tensor, gm_position, config)
    }

    fn write_contiguous_unchecked(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        positions: WritePositions,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;

        let mut output_elem = Line::empty(tile_size);

        #[unroll(unroll)]
        for i in 0..tile_size {
            output_elem[i] = results[positions.result + i];
        }

        out[positions.out / tile_size] = output_elem;
    }

    fn write_contiguous_checked(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        positions: WritePositions,
        _check_bounds: CheckBounds,
        _write_col: u32,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        // If vectorization matches, then it's certain to fit since tile_size divides block_sizes
        MatchingVectorization::write_contiguous_unchecked(out, results, positions, config)
    }
}

#[cube]
impl<N: Numeric> ContiguousAccess<N> for UnmatchingVectorization {
    fn read_contiguous_unchecked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N> {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;
        let line_size = tensor.line_size();
        let is_scalar = comptime!(line_size == 1);

        let mut vector = Line::empty(tile_size).fill(N::from_int(0));

        #[unroll(unroll)]
        for i in 0u32..comptime!(tile_size / line_size) {
            if comptime!(is_scalar) {
                vector[i] = tensor[gm_position + i][0];
            } else {
                let intermediate = tensor[gm_position + i];

                #[unroll(unroll)]
                for j in 0..comptime!(line_size) {
                    vector[i * line_size + j] = intermediate[j];
                }
            }
        }

        vector
    }

    fn read_contiguous_checked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        check_bounds: CheckBounds,
        read_info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N> {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;
        let line_size = tensor.line_size();
        let is_scalar = comptime!(line_size == 1);

        let mut vector = Line::empty(tile_size).fill(N::from_int(0));

        let mut num_loops = 0;
        if check_bounds.dim_horizontal > read_info.read_col {
            let num_reads = Min::min(check_bounds.dim_horizontal - read_info.read_col, tile_size);
            num_loops = num_reads / line_size;
        }

        for i in 0..num_loops {
            if comptime!(is_scalar) {
                vector[i] = tensor[gm_position + i][0];
            } else {
                let intermediate = tensor[gm_position + i];

                #[unroll(unroll)]
                for j in 0..comptime!(line_size) {
                    vector[i * line_size + j] = intermediate[j];
                }
            }
        }

        vector
    }

    fn write_contiguous_unchecked(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        positions: WritePositions,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;
        let line_size = out.line_size();
        let is_scalar = comptime!(line_size == 1);

        #[unroll(unroll)]
        for i in 0..comptime!(tile_size / line_size) {
            if comptime!(is_scalar) {
                out[i + positions.out] = Line::new(results[positions.result + i]);
            } else {
                let mut output_elem = Line::empty(line_size);

                #[unroll(unroll)]
                for j in 0..comptime!(line_size) {
                    let index = i * line_size + j;
                    output_elem[j] = results[positions.result + index];
                }

                out[i + positions.out / line_size] = output_elem;
            }
        }
    }

    fn write_contiguous_checked(
        out: &mut Tensor<Line<N>>,
        results: &Array<N>,
        positions: WritePositions,
        check_bounds: CheckBounds,
        write_col: u32,
        #[comptime] config: CubeTiling2dConfig,
    ) {
        let tile_size = config.tile_size;
        let line_size = out.line_size();
        let is_scalar = comptime!(line_size == 1);

        let mut num_loops = 0;
        if check_bounds.dim_horizontal > write_col {
            let num_writes = Min::min(check_bounds.dim_horizontal - write_col, tile_size);
            num_loops = num_writes / line_size;
        }

        for i in 0..num_loops {
            let unroll = config.unroll_tile;

            if comptime!(is_scalar) {
                out[i + positions.out] = Line::new(results[positions.result + i]);
            } else {
                let mut output_elem = Line::empty(line_size);

                #[unroll(unroll)]
                for j in 0u32..line_size {
                    let index = i * line_size + j;
                    output_elem[j] = results[positions.result + index];
                }

                out[i + positions.out / line_size] = output_elem;
            }
        }
    }
}

#[cube]
impl<N: Numeric> StridedAccess<N> for UnmatchingVectorization {
    fn read_strided_unchecked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        gm_stride: u32,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N> {
        let tile_size = config.tile_size;
        let unroll = config.unroll_tile;

        let mut vertical = Line::empty(tile_size);

        #[unroll(unroll)]
        for i in 0..tile_size {
            vertical[i] = tensor[gm_position + i * gm_stride][0];
        }

        vertical
    }

    fn read_strided_checked(
        tensor: &Tensor<Line<N>>,
        gm_position: u32,
        gm_stride: u32,
        check_bounds: CheckBounds,
        info: ReadTileInfo,
        #[comptime] config: CubeTiling2dConfig,
    ) -> Line<N> {
        let tile_size = config.tile_size;

        let mut vertical = Line::empty(tile_size);
        let mut num_reads = 0;

        let row = check_bounds.skip_row + info.read_row;
        let dim_vertical = check_bounds.dim_vertical;

        if dim_vertical > row {
            num_reads = Min::min(dim_vertical - row, tile_size);
        }

        for i in 0..num_reads {
            vertical[i] = tensor[gm_position + i * gm_stride][0];
        }

        for i in num_reads..tile_size {
            vertical[i] = N::from_int(0);
        }

        vertical
    }
}
