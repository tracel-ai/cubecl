//! Cell-major gather for expensive procedural filters separable over the contracted axes.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use super::super::super::registers;
use crate::*;

use super::base::{FactorReuse, GatherProblem};
use super::coords::{cell_position, offset_last};

/// Contract a separable procedural rhs, caching each factor's 1-D tap walk at its maximal reuse
/// level (block, row, column or cell).
#[cube]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric, V: Size, A: Size>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] problem: GatherProblem,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(problem.block.mr);
    let nr = comptime!(problem.block.nr);
    let cols = comptime!(problem.block.cols);
    let spread = comptime!(problem.block.spread);
    let aw = comptime!(problem.block.aw);
    let matrices = comptime!(problem.block.matrices());
    let batch_extents = comptime!(problem.block.batch_extents());

    let rhs_reader = rhs.nd_split::<ER, V>();
    let rhs_check = comptime!(rhs_reader.view.check);
    let factors = comptime!(
        problem
            .factors
            .expect("contract gather: the separable schedule needs a factorized lhs")
    );
    let kc = comptime!(problem.block.kc);
    let taps = comptime!(problem.taps);
    let caches_block = comptime!(problem.factor_reuse.contains(&FactorReuse::Block));
    let caches_column = comptime!(problem.factor_reuse.contains(&FactorReuse::Column));
    let factors_span_col = comptime!(
        problem
            .factor_reuse
            .iter()
            .any(|reuse| matches!(reuse, FactorReuse::Column | FactorReuse::Cell))
    );

    for mat in 0..matrices {
        let batch = Coords::constant(comptime!(batch_extents.clone())).unravel(mat.cast::<u32>());

        let mut acc = acc.matrix_accumulate::<E, A>(
            mat,
            comptime!(problem.block.acc_axes),
            comptime!(problem.block.space.clone()),
            comptime!(semiring.add()),
        );
        // Taps unroll on their own budget: each unrolled tap costs a body per cell.
        let acc_check = acc.check();
        let unroll =
            comptime!(problem.block.scalars() * kc <= config.budget && !rhs_check && !acc_check);
        let unroll_taps = comptime!(kc <= config.budget);
        let mut c = registers::seed::<E, V, A>(
            &mut acc,
            1usize,
            spread,
            aw,
            comptime!(mr),
            comptime!(nr),
            comptime!(cols),
            unroll,
        );

        let mut block_weights = Array::<EL>::new(taps);
        if comptime!(caches_block) {
            let block_anchor = rhs_anchor(
                &rhs_reader.map,
                &batch,
                0u32,
                0u32,
                comptime!(problem.clone()),
            );
            #[unroll]
            for f in 0..factors {
                if comptime!(problem.factor_reuse[f] == FactorReuse::Block) {
                    factor_walk::<EL>(
                        &mut block_weights,
                        comptime!(problem.offsets[f]),
                        lhs,
                        rhs,
                        &rhs_reader.map,
                        &block_anchor,
                        &batch,
                        0u32,
                        0u32,
                        f,
                        comptime!(problem.clone()),
                    );
                }
            }
        }

        let mut column_weights = Array::<EL>::new(taps * nr);
        if comptime!(caches_column) {
            #[unroll(unroll)]
            for n in 0..nr {
                let anchor = rhs_anchor(
                    &rhs_reader.map,
                    &batch,
                    0u32,
                    n as u32,
                    comptime!(problem.clone()),
                );
                #[unroll]
                for f in 0..factors {
                    if comptime!(problem.factor_reuse[f] == FactorReuse::Column) {
                        factor_walk::<EL>(
                            &mut column_weights,
                            n * comptime!(taps) + comptime!(problem.offsets[f]),
                            lhs,
                            rhs,
                            &rhs_reader.map,
                            &anchor,
                            &batch,
                            0u32,
                            n as u32,
                            f,
                            comptime!(problem.clone()),
                        );
                    }
                }
            }
        }

        #[unroll(unroll)]
        for i in 0..mr {
            let mut row_weights = Array::<EL>::new(taps);
            let row_anchor = rhs_anchor(
                &rhs_reader.map,
                &batch,
                i as u32,
                0u32,
                comptime!(problem.clone()),
            );
            #[unroll]
            for f in 0..factors {
                if comptime!(problem.factor_reuse[f] == FactorReuse::Row) {
                    factor_walk::<EL>(
                        &mut row_weights,
                        comptime!(problem.offsets[f]),
                        lhs,
                        rhs,
                        &rhs_reader.map,
                        &row_anchor,
                        &batch,
                        i as u32,
                        0u32,
                        f,
                        comptime!(problem.clone()),
                    );
                }
            }
            if comptime!(factors_span_col) {
                #[unroll(unroll)]
                for n in 0..nr {
                    let anchor = rhs_anchor(
                        &rhs_reader.map,
                        &batch,
                        i as u32,
                        n as u32,
                        comptime!(problem.clone()),
                    );
                    let mut cell_weights = Array::<EL>::new(taps);
                    #[unroll]
                    for f in 0..factors {
                        if comptime!(problem.factor_reuse[f] == FactorReuse::Cell) {
                            factor_walk::<EL>(
                                &mut cell_weights,
                                comptime!(problem.offsets[f]),
                                lhs,
                                rhs,
                                &rhs_reader.map,
                                &anchor,
                                &batch,
                                i as u32,
                                n as u32,
                                f,
                                comptime!(problem.clone()),
                            );
                        }
                    }

                    #[unroll(unroll_taps)]
                    for p in 0..kc {
                        let reduce_coords =
                            Coords::constant(comptime!(problem.block.reduce_extents.clone()))
                                .unravel(p.cast::<u32>());
                        let weight = tap_weight::<EL>(
                            &block_weights,
                            &row_weights,
                            &column_weights,
                            &cell_weights,
                            &reduce_coords,
                            n,
                            comptime!(factors),
                            comptime!(taps),
                            comptime!(problem.offsets.clone()),
                            comptime!(problem.factor_reuse.clone()),
                        );
                        let base = rhs_reader.map.advance(
                            &anchor,
                            cell_position(
                                &batch,
                                i as u32,
                                n as u32,
                                &reduce_coords,
                                comptime!(problem.rhs_space.clone()),
                                comptime!(problem.clone()),
                                comptime!(problem.block.vw),
                            ),
                            comptime!(problem.block.reduce.clone()),
                        );
                        let value = Vector::<E, V>::cast_from(rhs_reader.view.read(base));
                        // One semiring step, for the reason [`registers::rank1_update`] gives.
                        c[i * nr + n] = semiring.step::<Vector<E, V>>(
                            Vector::<E, V>::cast_from(weight),
                            value,
                            c[i * nr + n],
                        );
                    }
                }
            } else {
                // Only the taps move under this row, so the map's floors are taken once here.
                #[unroll(unroll_taps)]
                for p in 0..kc {
                    let reduce_coords =
                        Coords::constant(comptime!(problem.block.reduce_extents.clone()))
                            .unravel(p.cast::<u32>());
                    let weight = Vector::<E, V>::cast_from(row_cached_tap_weight::<EL>(
                        &block_weights,
                        &row_weights,
                        &reduce_coords,
                        comptime!(factors),
                        comptime!(problem.offsets.clone()),
                        comptime!(problem.factor_reuse.clone()),
                    ));

                    let base = rhs_reader.map.advance(
                        &row_anchor,
                        cell_position(
                            &batch,
                            i as u32,
                            0u32,
                            &reduce_coords,
                            comptime!(problem.rhs_space.clone()),
                            comptime!(problem.clone()),
                            comptime!(problem.block.vw),
                        ),
                        comptime!(problem.block.reduce.clone()),
                    );

                    #[unroll(unroll)]
                    for n in 0..nr {
                        let value = Vector::<E, V>::cast_from(rhs_reader.view.read(offset_last(
                            &base,
                            comptime!(rhs_reader.rank),
                            n as u32,
                        )));
                        // One semiring step, for the reason [`registers::rank1_update`] gives.
                        c[i * nr + n] = semiring.step::<Vector<E, V>>(weight, value, c[i * nr + n]);
                    }
                }
            }
        }
        registers::commit::<E, V, A>(
            &mut acc,
            c,
            1usize,
            spread,
            aw,
            comptime!(mr),
            comptime!(nr),
            comptime!(cols),
            unroll,
        );
    }
}

/// Anchor the rhs projection at one accumulator cell with every contracted coordinate at zero.
#[cube]
fn rhs_anchor(
    rhs_map: &ProjectionInKernel,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] problem: GatherProblem,
) -> CoordsDyn {
    rhs_map.anchor(
        cell_position(
            batch,
            row,
            col,
            &factor_coords(comptime!(problem.factors.unwrap()), 0usize, 0usize),
            comptime!(problem.rhs_space.clone()),
            comptime!(problem.clone()),
            comptime!(problem.block.vw),
        ),
        comptime!(problem.block.reduce.clone()),
    )
}

/// Evaluate one factor's complete 1-D tap walk at the accumulator coordinate where it is cached.
#[cube]
#[allow(clippy::too_many_arguments)]
fn factor_walk<EL: Numeric>(
    weights: &mut Array<EL>,
    offset: usize,
    lhs: &Tile,
    rhs: &Tile,
    rhs_map: &ProjectionInKernel,
    anchor: &CoordsDyn,
    batch: &Coords<u32>,
    row: u32,
    col: u32,
    #[comptime] factor: usize,
    #[comptime] problem: GatherProblem,
) {
    match comptime!(problem.normalization.clone()) {
        None =>
        {
            #[unroll]
            for k in 0..comptime!(problem.block.reduce_extents[factor]) {
                let pos = cell_position(
                    batch,
                    row,
                    col,
                    &factor_coords(comptime!(problem.factors.unwrap()), factor, k),
                    comptime!(problem.lhs_space.clone()),
                    comptime!(problem.clone()),
                    1usize,
                );
                weights[offset + comptime!(k)] = lhs.separable_factor::<EL>(pos, factor);
            }
        }
        Some(normalization) => {
            let mut sum = EL::from_int(0);
            #[unroll]
            for k in 0..comptime!(problem.block.reduce_extents[factor]) {
                let reduce_coords = factor_coords(comptime!(problem.factors.unwrap()), factor, k);
                let lhs_pos = cell_position(
                    batch,
                    row,
                    col,
                    &reduce_coords,
                    comptime!(problem.lhs_space.clone()),
                    comptime!(problem.clone()),
                    1usize,
                );
                let weight = lhs.separable_factor::<EL>(lhs_pos, factor);
                let weight = match comptime!(normalization.taps) {
                    TapSupport::InBounds => {
                        let rhs_pos = cell_position(
                            batch,
                            row,
                            col,
                            &reduce_coords,
                            comptime!(problem.rhs_space.clone()),
                            comptime!(problem.clone()),
                            comptime!(problem.block.vw),
                        );
                        let physical_pos = rhs_map.advance(
                            anchor,
                            rhs_pos,
                            comptime!(problem.block.reduce.clone()),
                        );
                        select(
                            rhs.separable_physical_tap_in_bounds(
                                &physical_pos,
                                comptime!(problem.block.reduce[factor]),
                            ),
                            weight,
                            EL::from_int(0),
                        )
                    }
                    TapSupport::Whole => weight,
                };
                weights[offset + comptime!(k)] = weight;
                sum += weight;
            }

            let reciprocal = guarded_recip::<EL>(sum, comptime!(normalization.guard));
            #[unroll]
            for k in 0..comptime!(problem.block.reduce_extents[factor]) {
                weights[offset + comptime!(k)] *= reciprocal;
            }
        }
    }
}

/// Multiply one tap's per-factor weights from their maximal cache levels.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tap_weight<EL: Numeric>(
    block: &Array<EL>,
    row: &Array<EL>,
    column: &Array<EL>,
    cell: &Array<EL>,
    reduce_coords: &Coords<u32>,
    column_index: usize,
    #[comptime] factors: usize,
    #[comptime] taps: usize,
    #[comptime] offsets: Vec<usize>,
    #[comptime] reuse: Vec<FactorReuse>,
) -> EL {
    let mut weight = factor_weight(
        block,
        row,
        column,
        cell,
        reduce_coords,
        column_index,
        0usize,
        taps,
        comptime!(offsets[0]),
        comptime!(reuse[0]),
    );
    #[unroll]
    for f in 1..factors {
        weight *= factor_weight(
            block,
            row,
            column,
            cell,
            reduce_coords,
            column_index,
            f,
            taps,
            comptime!(offsets[f]),
            comptime!(reuse[f]),
        );
    }
    weight
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn factor_weight<EL: Numeric>(
    block: &Array<EL>,
    row: &Array<EL>,
    column: &Array<EL>,
    cell: &Array<EL>,
    reduce_coords: &Coords<u32>,
    column_index: usize,
    #[comptime] factor: usize,
    #[comptime] taps: usize,
    #[comptime] offset: usize,
    #[comptime] reuse: FactorReuse,
) -> EL {
    let tap = factor_tap(reduce_coords, factor, offset);
    match comptime!(reuse) {
        FactorReuse::Block => block[tap],
        FactorReuse::Row => row[tap],
        FactorReuse::Column => column[column_index * comptime!(taps) + tap],
        FactorReuse::Cell => cell[tap],
    }
}

/// The column-free nesting only needs block- and row-cached factors.
#[cube]
fn row_cached_tap_weight<EL: Numeric>(
    block: &Array<EL>,
    row: &Array<EL>,
    reduce_coords: &Coords<u32>,
    #[comptime] factors: usize,
    #[comptime] offsets: Vec<usize>,
    #[comptime] reuse: Vec<FactorReuse>,
) -> EL {
    let first = factor_tap(reduce_coords, 0usize, comptime!(offsets[0]));
    let mut weight = match comptime!(reuse[0]) {
        FactorReuse::Block => block[first],
        FactorReuse::Row => row[first],
        FactorReuse::Column | FactorReuse::Cell => unreachable!(),
    };
    #[unroll]
    for f in 1..factors {
        let tap = factor_tap(reduce_coords, f, comptime!(offsets[f]));
        weight *= match comptime!(reuse[f]) {
            FactorReuse::Block => block[tap],
            FactorReuse::Row => row[tap],
            FactorReuse::Column | FactorReuse::Cell => unreachable!(),
        };
    }
    weight
}

#[cube]
fn factor_tap(
    reduce_coords: &Coords<u32>,
    #[comptime] factor: usize,
    #[comptime] offset: usize,
) -> usize {
    reduce_coords
        .at(factor)
        .plus(comptime!(offset as u32))
        .cast::<usize>()
}

#[cube]
fn factor_coords(
    #[comptime] factors: usize,
    #[comptime] factor: usize,
    #[comptime] tap: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();

    #[unroll]
    for f in 0..factors {
        out.push(comptime!(if f == factor { tap as u32 } else { 0u32 }));
    }

    out
}
