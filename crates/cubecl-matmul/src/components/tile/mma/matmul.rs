use std::marker::PhantomData;

use crate::components::MatrixLayout;
use crate::components::tile::io::{Filled, Strided, TileKind};
use crate::components::tile::mma::config::MmaMatmulConfig;
use crate::components::tile::{
    TileMatmul,
    mma::{reader::MmaStageReader, writer::MmaStageWriter},
};
use crate::components::tile::{mma::reader::MmaFragmentReader, tile_data::StridedTile};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, cmma::MmaDefinition, ir::MatrixIdent};

/// Uses one plane to perform a small matmul using accelerated instructions, with manual register
/// management.
/// Currently requires matrix layout to match the platform's preferred layout.
pub struct MmaMatmul<Lhs: TileKind = Strided, Rhs: TileKind = Strided, Acc: TileKind = Filled> {
    _ty: PhantomData<(Lhs, Rhs, Acc)>,
}

#[derive(CubeType)]
pub struct MmaFragment<E: Numeric> {
    fragment: Array<Line<E>>,
    #[cube(comptime)]
    layout: MatrixLayout,
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

    type LhsFragment = MmaFragment<L>;
    type RhsFragment = MmaFragment<R>;
    type AccFragment = MmaFragment<A>;

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
        let out_arr = def.execute(&lhs.fragment, &rhs.fragment, &out.fragment);
        let num_lines = def.lines_per_lane(MatrixIdent::Accumulator);

        #[unroll]
        for i in 0..num_lines {
            out.fragment[i] = out_arr[i];
        }
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let def = mma_definition::<L, R, A>(config);
        let line_size = def.line_size(MatrixIdent::A);
        let line_count = def.lines_per_lane(MatrixIdent::A);

        MmaFragment::<L> {
            fragment: Array::vectorized(line_count, line_size),
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let def = mma_definition::<L, R, A>(config);
        let line_size = def.line_size(MatrixIdent::B);
        let line_count = def.lines_per_lane(MatrixIdent::B);

        MmaFragment::<R> {
            fragment: Array::vectorized(line_count, line_size),
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let def = mma_definition::<L, R, A>(config);
        let line_size = def.line_size(MatrixIdent::Accumulator);
        let line_count = def.lines_per_lane(MatrixIdent::Accumulator);

        MmaFragment::<A> {
            fragment: Array::vectorized(line_count, line_size),
            layout,
        }
    }

    fn load_lhs<E: Numeric>(
        tile: &LhsTile::Tile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Self::LhsTile>::load_fragment(
            tile,
            &mut lhs.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::A,
            lhs.layout,
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
            &mut rhs.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::B,
            rhs.layout,
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
            &mut acc.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            acc.layout,
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
            &out.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            tile.layout,
            config,
        );
    }
}

#[cube]
pub(super) fn mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmulConfig,
) -> MmaDefinition<L, R, A> {
    let size = config.shared.tile_size;
    MmaDefinition::new(size.m(), size.n(), size.k())
}
