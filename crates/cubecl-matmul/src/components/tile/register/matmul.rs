use crate::components::tile::register::config::{ProductType, RegisterConfig};
use crate::components::tile::tile_data::Tile;
use crate::components::tile::{TileConfig, TileMatmul};
use crate::components::{MatmulPrecision, MatrixLayout, StageIdent};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

/// Uses one unit to perform a small matmul directly in registers
pub struct RegisterMatmul;

/// Doesn't impact performance much, but may increase kernel size too much when true (often ~6X).
///
/// TODO: make it configurable
static UNROLL: bool = false;

#[derive(CubeType)]
/// Contains the accumulated result in a row-major array of size rows x cols
pub struct TileAccumulator<EA: Numeric> {
    data: Array<EA>,
    #[cube(comptime)]
    rows: u32,
    #[cube(comptime)]
    cols: u32,
}

#[cube]
impl<MP: MatmulPrecision> TileMatmul<MP> for RegisterMatmul {
    type Config = RegisterConfig;
    type Lhs = Array<MP::ES>;
    type Rhs = Array<MP::ES>;
    type Accumulator = TileAccumulator<MP::EA>;

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        match config.product_type() {
            ProductType::Inner => Self::inner_product(lhs, rhs, acc, config),
            ProductType::Outer => Self::outer_product(lhs, rhs, acc, config),
        }
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        Array::new(config.tile_size().mk())
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        Array::new(config.tile_size().nk())
    }

    fn fill_lhs(tile: &Tile<MP::ES>, lhs: &mut Self::Lhs, #[comptime] config: Self::Config) {
        let size = config.tile_size();
        let lhs_line_size = config.stage_line_size(StageIdent::Lhs);
        let lhs_layout = config.matrix_layout(StageIdent::Lhs);

        match config.product_type() {
            ProductType::Inner => match lhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_plain(tile, lhs, size.m(), size.k(), lhs_line_size);
                }
                MatrixLayout::ColMajor => {
                    Self::fill_transposed(tile, lhs, size.k(), size.m(), lhs_line_size);
                }
            },
            ProductType::Outer => match lhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_transposed(tile, lhs, size.m(), size.k(), lhs_line_size);
                }
                MatrixLayout::ColMajor => {
                    Self::fill_plain(tile, lhs, size.k(), size.m(), lhs_line_size);
                }
            },
        }
    }

    fn fill_rhs(tile: &Tile<MP::ES>, rhs: &mut Self::Rhs, #[comptime] config: Self::Config) {
        let size = config.tile_size();
        let rhs_line_size = config.stage_line_size(StageIdent::Rhs);
        let rhs_layout = config.matrix_layout(StageIdent::Rhs);

        match config.product_type() {
            ProductType::Inner => match rhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_transposed(tile, rhs, size.k(), size.n(), rhs_line_size);
                }
                MatrixLayout::ColMajor => {
                    Self::fill_plain(tile, rhs, size.n(), size.k(), rhs_line_size);
                }
            },
            ProductType::Outer => match rhs_layout {
                MatrixLayout::RowMajor => {
                    Self::fill_plain(tile, rhs, size.k(), size.n(), rhs_line_size);
                }
                MatrixLayout::ColMajor => {
                    Self::fill_transposed(tile, rhs, size.n(), size.k(), rhs_line_size);
                }
            },
        }
    }

    fn fill_accumulator(
        tile: &Tile<MP::EA>,
        acc: &mut Self::Accumulator,
        #[comptime] _config: Self::Config,
    ) {
        #[unroll(UNROLL)]
        for i in 0..comptime!(acc.rows) {
            #[unroll(UNROLL)]
            for j in 0..comptime!(acc.cols) {
                acc.data[i * acc.cols + j] =
                    tile.slice.with_line_size(1u32)[i * tile.stride + j][0];
            }
        }
    }

    fn write_results(
        acc: &Self::Accumulator,
        slice: &mut SliceMut<Line<MP::EO>>,
        #[comptime] config: Self::Config,
    ) {
        let out_line_size = config.stage_line_size(StageIdent::Acc);
        #[unroll(UNROLL)]
        for i in 0..comptime!(acc.rows * acc.cols / out_line_size) {
            let mut line = Line::empty(out_line_size);
            #[unroll(UNROLL)]
            for j in 0..comptime!(out_line_size) {
                line[j] = acc.data[i * out_line_size + j];
            }
            slice[i] = Line::cast_from(line);
        }
    }

    fn allocate_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let rows = config.tile_size().m();
        let cols = config.tile_size().n();

        TileAccumulator::<MP::EA> {
            data: Array::<MP::EA>::new(rows * cols),
            rows,
            cols,
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] _config: Self::Config) {
        #[unroll(UNROLL)]
        for i in 0..comptime!(acc.rows * acc.cols) {
            acc.data[i] = MP::EA::cast_from(0);
        }
    }
}

#[cube]
impl RegisterMatmul {
    fn inner_product<ES: Numeric, EA: Numeric>(
        lhs: &Array<ES>,
        rhs: &Array<ES>,
        acc: &mut TileAccumulator<EA>,
        #[comptime] config: RegisterConfig,
    ) {
        let (m, n, k) =
            comptime! {let (m, n, k): (u32, u32, u32) = (*config.tile_size()).into(); (m, n, k)};

        #[unroll(UNROLL)]
        for m_ in 0..m {
            #[unroll(UNROLL)]
            for n_ in 0..n {
                #[unroll(UNROLL)]
                for k_ in 0..k {
                    let lhs_elem = EA::cast_from(lhs[m_ * k + k_]);
                    let rhs_elem = EA::cast_from(rhs[n_ * k + k_]);
                    acc.data[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn outer_product<ES: Numeric, EA: Numeric>(
        lhs: &Array<ES>,
        rhs: &Array<ES>,
        acc: &mut TileAccumulator<EA>,
        #[comptime] config: RegisterConfig,
    ) {
        let (m, n, k) =
            comptime! {let (m, n, k): (u32, u32, u32) = (*config.tile_size()).into(); (m, n, k)};

        #[unroll(UNROLL)]
        for k_ in 0..k {
            #[unroll(UNROLL)]
            for m_ in 0..m {
                let lhs_elem = EA::cast_from(lhs[k_ * m + m_]);
                #[unroll(UNROLL)]
                for n_ in 0..n {
                    let rhs_elem = EA::cast_from(rhs[k_ * n + n_]);
                    acc.data[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn fill_plain<ES: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ES>,
        #[comptime] num_segments: u32,
        #[comptime] segment_size: u32,
        #[comptime] line_size: u32,
    ) {
        let num_lines_per_segment = segment_size / line_size;

        #[unroll(UNROLL)]
        for segment in 0..num_segments {
            #[unroll(UNROLL)]
            for line_within_segment in 0..num_lines_per_segment {
                let line = tile.get_line(segment, line_within_segment);
                #[unroll(UNROLL)]
                for pos_within_line in 0..line_size {
                    array[segment * segment_size
                        + line_within_segment * line_size
                        + pos_within_line] = line[pos_within_line];
                }
            }
        }
    }

    fn fill_transposed<ES: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ES>,
        #[comptime] num_segments: u32,
        #[comptime] segment_size: u32,
        #[comptime] line_size: u32,
    ) {
        let num_lines_per_segment = segment_size / line_size;

        #[unroll(UNROLL)]
        for segment in 0..num_segments {
            #[unroll(UNROLL)]
            for line_within_segment in 0..num_lines_per_segment {
                let line = tile.get_line(segment, line_within_segment);
                #[unroll(UNROLL)]
                for pos_within_line in 0..line_size {
                    array[(line_within_segment * line_size + pos_within_line) * num_segments
                        + segment] = line[pos_within_line];
                }
            }
        }
    }
}
