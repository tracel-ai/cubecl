use std::marker::PhantomData;

use crate::components::tile::{
    SharedTileConfig, TileMatmul,
    plane_vec_mat_inner_product::{reader::MatrixFragmentReader, writer::MatrixStageWriter},
};
use crate::components::tile::{
    io::Strided, plane_vec_mat_inner_product::reader::MatrixStageReader, tile_data::StridedTile,
};
use crate::components::tile::{
    io::TileKind, plane_vec_mat_inner_product::reader::VectorStageReader,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

/// Uses one unit to perform a small matmul directly in registers
pub struct PlaneVecMatInnerProduct<Acc: TileKind> {
    _ty: PhantomData<Acc>,
}

#[derive(CubeType)]
pub struct LineContainer<E: Numeric> {
    pub line: Line<E>,
}

#[cube]
impl<E: Numeric> LineContainer<E> {
    fn new(#[comptime] size: u32) -> LineContainer<E> {
        LineContainer::<E> {
            line: Line::empty(size),
        }
    }
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric, AccTile: TileKind> TileMatmul<L, R, A>
    for PlaneVecMatInnerProduct<AccTile>
where
    MatrixStageReader<AccTile>: MatrixFragmentReader<TileKind = AccTile>,
{
    type Config = SharedTileConfig;

    // One line per unit in the plane
    type LhsFragment = LineContainer<L>;
    // For each n: one line per unit in the plane
    type RhsFragment = Sequence<LineContainer<R>>;

    // For each n: one line stored at unit pos 0, that will be reduced to a scalar only when writing at the end
    type AccFragment = Sequence<LineContainer<A>>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: SharedTileConfig,
    ) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.tile_size.n() {
            let lhs: Line<A> = Line::cast_from(lhs.line);
            let rhs: Line<A> = Line::cast_from(rhs.index(n).line);

            plane_sum_lined(lhs * rhs, acc.index_mut(n), config.lhs_stage_line_size);

            comptime![n += 1];
        }
    }

    fn allocate_lhs(#[comptime] config: SharedTileConfig) -> Self::LhsFragment {
        LineContainer::<L>::new(config.lhs_stage_line_size)
    }

    fn allocate_rhs(#[comptime] config: SharedTileConfig) -> Self::RhsFragment {
        let mut rhs = Sequence::new();
        #[unroll]
        for _ in 0..config.tile_size.n() {
            rhs.push(LineContainer::new(config.lhs_stage_line_size))
        }
        rhs
    }

    fn allocate_acc(#[comptime] config: SharedTileConfig) -> Self::AccFragment {
        let mut acc = Sequence::new();
        #[unroll]
        for _ in 0..config.tile_size.n() {
            acc.push(LineContainer::new(config.lhs_stage_line_size))
        }
        acc
    }

    fn load_lhs<E: Numeric>(
        tile: &StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: SharedTileConfig,
    ) {
        VectorStageReader::load_fragment(tile, lhs)
    }

    fn load_rhs<E: Numeric>(
        tile: &StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: SharedTileConfig,
    ) {
        MatrixStageReader::<Strided>::load_fragment(tile, rhs, config)
    }

    fn load_acc<E: Numeric>(
        tile: &AccTile::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: SharedTileConfig,
    ) {
        MatrixStageReader::<AccTile>::load_fragment(tile, acc, config);
    }

    fn write_results<E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &Self::AccFragment,
        #[comptime] config: SharedTileConfig,
    ) {
        MatrixStageWriter::store_fragment(tile, acc, config)
    }
}

#[cube]
fn plane_sum_lined<E: Numeric>(
    line_to_sum: Line<E>,
    line_accumulator: &mut LineContainer<E>,
    #[comptime] line_size: u32,
) {
    let mut line_iterator = comptime![0];

    #[unroll]
    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..line_size {
        line_accumulator.line[line_iterator] += plane_sum(line_to_sum[line_iterator]);

        comptime![line_iterator += 1];
    }
}
