//! The general N-D gather nest, K-major with each operand read hoisted to its widest reuse scope.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use super::super::super::registers;
use crate::*;

use super::base::{GatherProblem, LhsRole, RhsRole};
use super::coords::cell_read;

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and block's, `A` the accumulator's.
#[cube]
pub(super) fn nest<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] problem: GatherProblem,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let matrices = comptime!(problem.block.matrices());
    let batch_extents = comptime!(problem.block.batch_extents());

    let lhs_view = lhs.nd_packed::<EL, L>(comptime!(Guard::Checked));
    let rhs_view = rhs.nd_packed::<ER, V>(comptime!(Guard::Checked));
    // `comptime!`-bound so `#[unroll(flag)]` sees a comptime flag; otherwise it silently rolls.
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);

    let eligible = comptime!(problem.block.scalars() <= config.budget);
    // A clamp guard can't be retired by the corner check; see `Tile::guard_provable`.
    let lhs_provable = lhs.guard_provable();
    let rhs_provable = rhs.guard_provable();
    let provable = comptime!(lhs_provable && rhs_provable);
    // A spread block's last column reads past the proven box, so keep the leaf checked.
    let spread_overhang = comptime!(registers::spread_guard(
        problem.block.spread,
        problem.block.cols
    ));
    let split_operands = comptime!(
        config.split_edge && eligible && provable && !spread_overhang && (lhs_check || rhs_check)
    );
    let operands_inside = if comptime!(split_operands) {
        box_in_bounds::<EL, L>(
            &lhs_view,
            comptime!(problem.lhs_space.clone()),
            comptime!(problem.block.lw),
        ) && box_in_bounds::<ER, V>(
            &rhs_view,
            comptime!(problem.rhs_space.clone()),
            comptime!(problem.block.vw),
        )
    } else {
        comptime!(false).runtime()
    };

    for mat in 0..matrices {
        let batch = Coords::constant(comptime!(batch_extents.clone())).unravel(mat.cast::<u32>());

        let mut acc = acc.matrix_accumulate::<E, A>(
            mat,
            comptime!(problem.block.acc_axes),
            comptime!(problem.block.space.clone()),
            comptime!(semiring.add()),
        );

        let acc_check = acc.check();
        let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);

        // Interior instances prove the operand box once and read unguarded. The accumulator stays
        // guarded on both sides: it must never write outside the output.
        if comptime!(split_operands) {
            let inside = operands_inside;
            if inside {
                walk::<E, EL, L, ER, V, A>(
                    &lhs.nd_packed::<EL, L>(comptime!(Guard::Proved)),
                    &rhs.nd_packed::<ER, V>(comptime!(Guard::Proved)),
                    &mut acc,
                    &batch,
                    comptime!(problem.clone()),
                    config,
                    comptime!(true),
                    semiring,
                );
            } else {
                walk::<E, EL, L, ER, V, A>(
                    &lhs_view,
                    &rhs_view,
                    &mut acc,
                    &batch,
                    comptime!(problem.clone()),
                    config,
                    comptime!(false),
                    semiring,
                );
            }
        } else {
            walk::<E, EL, L, ER, V, A>(
                &lhs_view,
                &rhs_view,
                &mut acc,
                &batch,
                comptime!(problem.clone()),
                config,
                unroll,
                semiring,
            );
        }
    }
}

/// Whether every read the walk takes through `view` lands inside it, proven at the box's two
/// extreme corners. Does not cover columns overhanging the extent (a spread block's `nr`).
#[cube]
#[allow(clippy::needless_range_loop)]
fn box_in_bounds<T: Numeric, W: Size>(
    view: &Masked<'_, Vector<T, W>, CoordsDyn>,
    #[comptime] space: Space,
    #[comptime] width: usize,
) -> bool {
    let rank = comptime!(space.rank());
    let extents = comptime!(crate::line_extents(&space, width, 0, rank));

    let mut near = CoordsDyn::new();
    let mut far = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        near.push(0u32.runtime());
        far.push(comptime!(extents[p] as u32 - 1).runtime());
    }

    view.is_in_bounds(near) && view.is_in_bounds(far)
}

/// The `K` walk over one batch matrix: seed the block, sum `kc` rank-1 updates, commit it back.
#[cube]
#[allow(clippy::too_many_arguments)]
fn walk<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &Masked<'_, Vector<ER, V>, CoordsDyn>,
    acc: &mut AccumulateView<'_, E, A>,
    batch: &Coords<u32>,
    #[comptime] problem: GatherProblem,
    #[comptime] config: RegisterBlock,
    #[comptime] unroll: bool,
    #[comptime] semiring: Semiring,
) {
    // Comptime locals: `#[unroll(flag)]` and `0..n` bounds silently roll otherwise.
    let mr = comptime!(problem.block.mr);
    let nr = comptime!(problem.block.nr);
    let cols = comptime!(problem.block.cols);
    let contracted_per_step = comptime!(problem.block.contracted_per_step);
    let spread = comptime!(problem.block.spread);
    let aw = comptime!(problem.block.aw);
    let lw = comptime!(problem.block.lw);
    let kc = comptime!(problem.block.kc);

    // The last physical line can be partial, so keep a short tail.
    let k_lines = comptime!(kc / lw);
    let k_tail = comptime!(kc % lw);
    // A col-lined lhs has no `K` component to extract, so the fan-out buys nothing.
    let component_fanout = comptime!(
        config.component_fanout
            && problem.lhs != LhsRole::LinedAlongColumn
            && problem.block.component_index_exact()
    );

    let mut c = registers::seed::<E, V, A>(
        acc,
        contracted_per_step,
        spread,
        aw,
        comptime!(mr),
        comptime!(nr),
        comptime!(cols),
        unroll,
    );

    // One rhs line per column, reused by every row; allocated once for the whole K walk.
    let mut b = Array::<Vector<E, V>>::new(comptime!(nr));

    if comptime!(contracted_per_step > 1) {
        for step in 0..comptime!(kc / contracted_per_step) {
            rank1_update::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut c,
                &mut b,
                batch,
                step * comptime!(contracted_per_step),
                comptime!(None),
                unroll,
                comptime!(problem.clone()),
                semiring,
            );
        }
    } else if comptime!(component_fanout && lw > 1) {
        for line in 0..k_lines {
            #[unroll]
            for unit in 0..lw {
                rank1_update::<E, EL, L, ER, V>(
                    lhs_view,
                    rhs_view,
                    &mut c,
                    &mut b,
                    batch,
                    line * lw + unit,
                    comptime!(Some(unit)),
                    unroll,
                    comptime!(problem.clone()),
                    semiring,
                );
            }
        }
        #[unroll]
        for unit in 0..k_tail {
            rank1_update::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut c,
                &mut b,
                batch,
                comptime!(k_lines * lw + unit),
                comptime!(Some(unit)),
                unroll,
                comptime!(problem.clone()),
                semiring,
            );
        }
    } else {
        // CPU and scalar lines keep the flat walk, avoiding a wide fan-out body in LLVM IR.
        // A small register block can still contract a large filter (e.g. a depthwise weight
        // gradient). Bound tap expansion separately while keeping the cells in registers.
        // See https://github.com/tracel-ai/burn/issues/6054.
        let unroll_taps = comptime!(unroll && kc <= config.budget);
        #[unroll(unroll_taps)]
        for p in 0..kc {
            rank1_update::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut c,
                &mut b,
                batch,
                p,
                comptime!(None),
                unroll,
                comptime!(problem.clone()),
                semiring,
            );
        }
    }

    registers::commit::<E, V, A>(
        acc,
        c,
        contracted_per_step,
        spread,
        aw,
        comptime!(mr),
        comptime!(nr),
        comptime!(cols),
        unroll,
    );
}

/// One gathered rank-1 update, each read hoisted to the coarsest cell its operand is invariant
/// over. `unit` is the fixed component to extract; `None` resolves it from `reduce_coords`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &Masked<'_, Vector<ER, V>, CoordsDyn>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    batch: &Coords<u32>,
    p: usize,
    #[comptime] unit: Option<usize>,
    #[comptime] unroll: bool,
    #[comptime] problem: GatherProblem,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(problem.block.mr);
    let nr = comptime!(problem.block.nr);
    let contracted_per_step = comptime!(problem.block.contracted_per_step);
    let lw = comptime!(problem.block.lw);
    let reduce_coords =
        Coords::constant(comptime!(problem.block.reduce_extents.clone())).unravel(p.cast::<u32>());

    let k_axis_idx = comptime!(problem.block.reduce.len() - 1);
    if comptime!(problem.rhs == RhsRole::FreeOfRow) {
        #[unroll(unroll)]
        for n in 0..nr {
            b[n] = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                0u32,
                n as u32,
                &reduce_coords,
                comptime!(problem.rhs_space.clone()),
                comptime!(problem.clone()),
                comptime!(problem.block.vw),
            ));
        }
    }
    #[unroll(unroll)]
    for i in 0..mr {
        // Row-invariant reads, taken once; zero and evaluated away when the cell loop reads them.
        let mut a_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(problem.lhs == LhsRole::FreeOfColumn) {
            // Same position for every component of one line.
            let line = cell_read::<EL, L>(
                lhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(problem.lhs_space.clone()),
                comptime!(problem.clone()),
                lw,
            );
            a_row = line_component::<E, EL, L, V>(
                line,
                &reduce_coords,
                unit,
                contracted_per_step,
                lw,
                k_axis_idx,
            );
        }
        let mut b_row = Vector::<E, V>::cast_from(E::from_int(0));
        if comptime!(problem.rhs == RhsRole::PerRow) {
            b_row = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                i as u32,
                0u32,
                &reduce_coords,
                comptime!(problem.rhs_space.clone()),
                comptime!(problem.clone()),
                comptime!(problem.block.vw),
            ));
        }
        #[unroll(unroll)]
        for n in 0..nr {
            let a = if comptime!(problem.lhs != LhsRole::FreeOfColumn) {
                // A col-lined lhs's line is the cell: each column is a different value.
                let line = cell_read::<EL, L>(
                    lhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(problem.lhs_space.clone()),
                    comptime!(problem.clone()),
                    lw,
                );
                if comptime!(problem.lhs == LhsRole::LinedAlongColumn) {
                    Vector::<E, V>::cast_from(line)
                } else {
                    line_component::<E, EL, L, V>(
                        line,
                        &reduce_coords,
                        unit,
                        contracted_per_step,
                        lw,
                        k_axis_idx,
                    )
                }
            } else {
                a_row
            };
            let v = if comptime!(problem.rhs == RhsRole::FreeOfRow) {
                b[n]
            } else if comptime!(problem.rhs == RhsRole::PerRow) {
                b_row
            } else {
                Vector::<E, V>::cast_from(cell_read::<ER, V>(
                    rhs_view,
                    batch,
                    i as u32,
                    n as u32,
                    &reduce_coords,
                    comptime!(problem.rhs_space.clone()),
                    comptime!(problem.clone()),
                    comptime!(problem.block.vw),
                ))
            };
            // One semiring step, for the reason [`registers::rank1_update`] gives.
            c[i * nr + n] = semiring.step::<Vector<E, V>>(a, v, c[i * nr + n]);
        }
    }
}

/// The `K` component of one lhs line, widened into the accumulate element.
#[cube]
fn line_component<E: Numeric, EL: Numeric, L: Size, V: Size>(
    line: Vector<EL, L>,
    reduce_coords: &Coords<u32>,
    #[comptime] unit: Option<usize>,
    #[comptime] contracted_per_step: usize,
    #[comptime] lw: usize,
    #[comptime] k_axis_idx: usize,
) -> Vector<E, V> {
    if comptime!(contracted_per_step > 1) {
        Vector::<E, V>::cast_from(line)
    } else if comptime!(unit.is_some()) {
        Vector::<E, V>::cast_from(line.extract(comptime!(unit.unwrap())))
    } else if comptime!(lw == 1) {
        Vector::<E, V>::cast_from(line.extract(0usize))
    } else {
        let last_k = reduce_coords.at(comptime!(k_axis_idx));
        Vector::<E, V>::cast_from(
            line.extract_dynamic((last_k % comptime!(lw as u32)).cast::<usize>()),
        )
    }
}
