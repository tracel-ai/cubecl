use std::marker::PhantomData;

use crate::components::StageIdent;
use crate::components::tile::{
    TileConfig, TileMatmul, plane_vec_mat_inner_product::reader::MatrixFragmentReader,
};
use crate::components::tile::{
    plane_vec_mat_inner_product::reader::MatrixTileReader, reader::Strided, tile_data::StridedTile,
};
use crate::components::tile::{
    plane_vec_mat_inner_product::{
        config::PlaneVecMatInnerProductConfig, reader::VectorTileReader,
    },
    reader::TileKind,
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
impl<L: Numeric, R: Numeric, A: Numeric, Acc: TileKind> TileMatmul<L, R, A>
    for PlaneVecMatInnerProduct<Acc>
where
    MatrixTileReader<Acc>: MatrixFragmentReader<TileKind = Acc>,
{
    type Config = PlaneVecMatInnerProductConfig;

    // One line per unit in the plane
    type LhsFragment = LineContainer<L>;
    // For each n: one line per unit in the plane
    type RhsFragment = Sequence<LineContainer<R>>;

    // For each n: one line stored at unit pos 0, that will be reduced to a scalar only when writing at the end
    type AccFragment = Sequence<LineContainer<A>>;

    type LhsTileReader = VectorTileReader;
    type RhsTileReader = MatrixTileReader<Strided>;
    type AccTileReader = MatrixTileReader<Acc>;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let mut n = comptime![0];

        #[unroll]
        #[allow(clippy::explicit_counter_loop)]
        for _ in 0..config.n() {
            let lhs: Line<A> = Line::cast_from(lhs.line);
            let rhs: Line<A> = Line::cast_from(rhs.index(n).line);

            plane_sum_lined(lhs * rhs, acc.index_mut(n), config.reduce_line_size());

            comptime![n += 1];
        }
    }

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::LhsFragment {
        LineContainer::<L>::new(config.reduce_line_size())
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::RhsFragment {
        let mut rhs = Sequence::new();
        #[unroll]
        for _ in 0..config.n() {
            rhs.push(LineContainer::new(config.reduce_line_size()))
        }
        rhs
    }

    fn allocate_acc(#[comptime] config: Self::Config) -> Self::AccFragment {
        let mut acc = Sequence::new();
        #[unroll]
        for _ in 0..config.n() {
            acc.push(LineContainer::new(config.reduce_line_size()))
        }
        acc
    }

    fn load_lhs<E: Numeric>(
        tile: StridedTile<E>,
        lhs: &mut Self::LhsFragment,
        #[comptime] _config: Self::Config,
    ) {
        Self::LhsTileReader::load_fragment(tile, lhs)
    }

    fn load_rhs<E: Numeric>(
        tile: StridedTile<E>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::RhsTileReader::load_fragment(tile, rhs, config)
    }

    fn load_acc<E: Numeric>(
        tile: Acc::Tile<E>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        Self::AccTileReader::load_fragment(tile, acc, config);
    }

    fn write_results<E: Numeric>(
        acc: &Self::AccFragment,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        if UNIT_POS_X == 0 {
            let out_line_size = config.stage_line_size(StageIdent::Acc);
            let total_out_lines = config.n() / out_line_size;
            let mut out_line_iter = comptime![0];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..total_out_lines {
                let mut out_line = Line::<E>::empty(out_line_size);
                let mut within_line = comptime![0];

                #[unroll]
                #[allow(clippy::explicit_counter_loop)]
                for _ in 0..out_line_size {
                    let n = comptime!(out_line_iter * out_line_size + within_line);

                    let line_container = acc.index(n);
                    let mut sum = A::from_int(0);
                    for i in 0..config.reduce_line_size() {
                        sum += line_container.line[i];
                    }

                    out_line[within_line] = E::cast_from(sum);
                    comptime![within_line += 1];
                }

                slice[out_line_iter] = out_line;
                comptime![out_line_iter += 1];
            }
        }
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
