use std::marker::PhantomData;

use crate::components::StageIdent;
use crate::components::tile::{TileConfig, TileMatmul, register::reader::RegisterFragmentReader};
use crate::components::tile::{
    reader::Strided,
    register::{
        config::{ProductType, RegisterConfig},
        reader::RegisterTileReader,
    },
};
use crate::components::tile::{reader::TileKind, tile_data::Tile};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

/// Uses one unit to perform a small matmul directly in registers
pub struct RegisterMatmul<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

/// Doesn't impact performance much, but may increase kernel size too much when true (often ~6X).
///
/// TODO: make it configurable
static UNROLL: bool = false;

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, Acc: TileKind> TileMatmul<L, R, A> for RegisterMatmul<Acc>
where
    RegisterTileReader<Acc>: RegisterFragmentReader<TileKind = Acc>,
{
    type Config = RegisterConfig;
    type LhsFragment = Array<L>;
    type RhsFragment = Array<R>;
    type AccFragment = Array<A>;

    type LhsTileReader = RegisterTileReader<Strided>;
    type RhsTileReader = RegisterTileReader<Strided>;
    type AccTileReader = RegisterTileReader<Acc>;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        match config.product_type() {
            ProductType::Inner => Self::inner_product(lhs, rhs, acc, config),
            ProductType::Outer => Self::outer_product(lhs, rhs, acc, config),
        }
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::LhsFragment {
        Array::new(config.tile_size().mk())
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::RhsFragment {
        Array::new(config.tile_size().nk())
    }

    fn allocate_acc(#[comptime] config: Self::Config) -> Self::AccFragment {
        Array::new(config.tile_size().mn())
    }

    fn load_lhs<E: Numeric>(
        tile: Tile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::LhsTileReader::load_fragment(tile, lhs, StageIdent::Lhs, config)
    }

    fn load_rhs<E: Numeric>(
        tile: Tile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::RhsTileReader::load_fragment(tile, rhs, StageIdent::Rhs, config)
    }

    fn load_acc<E: Numeric>(
        tile: Acc::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::AccTileReader::load_fragment(tile, acc, StageIdent::Acc, config);
    }

    fn write_results<E: Numeric>(
        acc: &Self::AccFragment,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        let out_line_size = config.stage_line_size(StageIdent::Acc);
        #[unroll(UNROLL)]
        for i in 0..comptime!(config.tile_size().mn() / out_line_size) {
            let mut line = Line::empty(out_line_size);
            #[unroll(UNROLL)]
            for j in 0..comptime!(out_line_size) {
                line[j] = acc[i * out_line_size + j];
            }
            slice[i] = Line::cast_from(line);
        }
    }
}

#[cube]
impl<Acc: TileKind> RegisterMatmul<Acc> {
    fn inner_product<Lhs: Numeric, Rhs: Numeric, EA: Numeric>(
        lhs: &Array<Lhs>,
        rhs: &Array<Rhs>,
        acc: &mut Array<EA>,
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
                    acc[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    fn outer_product<Lhs: Numeric, Rhs: Numeric, EA: Numeric>(
        lhs: &Array<Lhs>,
        rhs: &Array<Rhs>,
        acc: &mut Array<EA>,
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
                    acc[m_ * n + n_] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    pub fn load_plain<ES: Numeric, ER: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ER>,
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
                        + pos_within_line] = ER::cast_from(line[pos_within_line]);
                }
            }
        }
    }

    pub fn load_transposed<ES: Numeric, ER: Numeric>(
        tile: &Tile<ES>,
        array: &mut Array<ER>,
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
                        + segment] = ER::cast_from(line[pos_within_line]);
                }
            }
        }
    }
}
