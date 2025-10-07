use std::marker::PhantomData;

use crate::components::tile::{TileConfig, TileMatmul, register::reader::RegisterFragmentReader};
use crate::components::tile::{
    io::Strided,
    register::{
        config::{ProductType, RegisterConfig},
        reader::RegisterStageReader,
    },
};
use crate::components::tile::{io::TileKind, tile_data::StridedTile};
use crate::components::{StageIdent, tile::register::writer::RegisterStageWriter};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

/// Uses one unit to perform a small matmul directly in registers
pub struct RegisterMatmul<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

/// Doesn't impact performance much, but may increase kernel size too much when true (often ~6X).
///
/// TODO: make it configurable
pub(super) const UNROLL: bool = false;

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, AccTile: TileKind> TileMatmul<L, R, A>
    for RegisterMatmul<AccTile>
where
    RegisterStageReader<AccTile>: RegisterFragmentReader<TileKind = AccTile>,
{
    type Config = RegisterConfig;
    type LhsFragment = Array<L>;
    type RhsFragment = Array<R>;
    type AccFragment = Array<A>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = AccTile;
    type OutTile = Strided;

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
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageReader::<Strided>::load_fragment(tile, lhs, StageIdent::Lhs, config)
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageReader::<Strided>::load_fragment(tile, rhs, StageIdent::Rhs, config)
    }

    fn load_acc<E: Numeric>(
        tile: &AccTile::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageReader::<AccTile>::load_fragment(tile, acc, StageIdent::Acc, config);
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageWriter::store_fragment(tile, acc, config)
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
        tile: &StridedTile<ES>,
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
        tile: &StridedTile<ES>,
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
