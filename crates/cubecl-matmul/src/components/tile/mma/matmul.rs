use std::marker::PhantomData;

use crate::components::tile::io::{Filled, Strided, TileKind};
use crate::components::tile::{
    TileConfig, TileMatmul,
    mma::{reader::MmaStageReader, writer::MmaStageWriter},
};
use crate::components::tile::{mma::reader::MmaFragmentReader, tile_data::StridedTile};
use crate::components::{StageIdent, tile::mma::config::MmaMatmulConfig};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, cmma::MmaDefinition, ir::MatrixIdent};

/// Uses one plane to perform a small matmul using accelerated instructions, with manual register
/// management.
/// Currently requires matrix layout to match the platform's preferred layout.
pub struct MmaMatmul<Lhs: TileKind = Strided, Rhs: TileKind = Strided, Acc: TileKind = Filled> {
    _ty: PhantomData<(Lhs, Rhs, Acc)>,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, LhsTile: TileKind, RhsTile: TileKind, AccTile: TileKind>
    TileMatmul<L, R, A> for MmaMatmul<LhsTile, RhsTile, AccTile>
where
    MmaStageReader<LhsTile>: MmaFragmentReader<TileKind = LhsTile>,
    MmaStageReader<RhsTile>: MmaFragmentReader<TileKind = RhsTile>,
    MmaStageReader<AccTile>: MmaFragmentReader<TileKind = AccTile>,
{
    type Config = MmaMatmulConfig;
    type LhsFragment = Array<Line<L>>;
    type RhsFragment = Array<Line<R>>;
    type AccFragment = Array<Line<A>>;

    type LhsTile = LhsTile;
    type RhsTile = RhsTile;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        out: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let def = mma_definition(config);
        let out_arr = def.execute(lhs, rhs, out);
        let num_lines = def.lines_per_lane(MatrixIdent::Accumulator);

        #[unroll]
        for i in 0..num_lines {
            out[i] = out_arr[i];
        }
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::LhsFragment {
        let def = mma_definition::<L, R, A>(config);
        let line_size = def.line_size(MatrixIdent::A);
        let line_count = def.lines_per_lane(MatrixIdent::A);
        Array::vectorized(line_count, line_size)
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::RhsFragment {
        let def = mma_definition::<L, R, A>(config);
        let line_size = def.line_size(MatrixIdent::B);
        let line_count = def.lines_per_lane(MatrixIdent::B);
        Array::vectorized(line_count, line_size)
    }

    fn allocate_acc(#[comptime] config: Self::Config) -> Self::AccFragment {
        let def = mma_definition::<L, R, A>(config);
        let line_size = def.line_size(MatrixIdent::Accumulator);
        let line_count = def.lines_per_lane(MatrixIdent::Accumulator);
        Array::vectorized(line_count, line_size)
    }

    fn load_lhs<E: Numeric>(
        tile: &LhsTile::Tile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Self::LhsTile>::load_fragment(
            tile,
            lhs,
            mma_definition::<L, R, A>(config),
            MatrixIdent::A,
            config.matrix_layout(StageIdent::Lhs),
            config,
        );
    }

    fn load_rhs<E: Numeric>(
        tile: &RhsTile::Tile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Self::RhsTile>::load_fragment(
            tile,
            rhs,
            mma_definition::<L, R, A>(config),
            MatrixIdent::B,
            config.matrix_layout(StageIdent::Rhs),
            config,
        );
    }

    fn load_acc<E: Numeric>(
        tile: &AccTile::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Self::AccTile>::load_fragment(
            tile,
            acc,
            mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            config.matrix_layout(StageIdent::Acc),
            config,
        );
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        out: &Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageWriter::store_fragment(
            tile,
            out,
            mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            config.matrix_layout(StageIdent::Out),
            config,
        );
    }
}

#[cube]
pub(super) fn mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmulConfig,
) -> MmaDefinition<L, R, A> {
    let size = config.tile_size();
    MmaDefinition::new(size.m(), size.n(), size.k())
}
