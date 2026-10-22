//! The `mr × nr` register block: seed it from the accumulator, contract into it, commit it back.

use cubecl::prelude::*;

use crate::*;

/// `c += lhs · rhs` over the block, one line of the contraction at a time, each factor times its
/// scales.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, RL: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    lhs_scales: &FactorReader,
    rhs: &MatrixView<'_, Vector<ER, RL>>,
    rhs_scales: &FactorReader,
    c: &mut Array<Vector<E, V>>,
    #[comptime] lw: usize,
    #[comptime] contracted_per_step: usize,
    #[comptime] along_k: bool,
    #[comptime] columns: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] component_fanout: bool,
    #[comptime] semiring: Semiring,
) {
    comptime!(assert!(
        columns == 1 || along_k,
        "mm: a rhs load holds the runs of {columns} columns along the contraction, and this rhs \
         lines along the accumulator"
    ));
    let mut b = Array::<Vector<E, V>>::new(nr);
    let folded = along_k;
    let width = comptime!(match folded {
        true => contracted_per_step,
        false => lw,
    });
    let lines = comptime!(kc / width);
    let tail = comptime!(kc % width);
    // Constant units let the backend fold an `mr`-row fan-out's repeated line reads into one.
    let fixed = comptime!(folded || (component_fanout && lw > 1) || lw == 1);

    for line in 0..lines {
        if comptime!(folded) {
            rank1_update::<E, EL, L, ER, V, RL>(
                lhs,
                lhs_scales,
                rhs,
                rhs_scales,
                c,
                &mut b,
                0usize,
                line as u32,
                0usize,
                comptime!(Some(0usize)),
                contracted_per_step,
                along_k,
                columns,
                mr,
                nr,
                unroll,
                semiring,
            );
        } else if comptime!(fixed) {
            #[unroll]
            for unit in 0..lw {
                rank1_update::<E, EL, L, ER, V, RL>(
                    lhs,
                    lhs_scales,
                    rhs,
                    rhs_scales,
                    c,
                    &mut b,
                    line * lw + unit,
                    line as u32,
                    0usize,
                    comptime!(Some(unit)),
                    contracted_per_step,
                    along_k,
                    columns,
                    mr,
                    nr,
                    unroll,
                    semiring,
                );
            }
        } else {
            for unit in 0..lw {
                rank1_update::<E, EL, L, ER, V, RL>(
                    lhs,
                    lhs_scales,
                    rhs,
                    rhs_scales,
                    c,
                    &mut b,
                    line * lw + unit,
                    line as u32,
                    unit,
                    comptime!(None),
                    contracted_per_step,
                    along_k,
                    columns,
                    mr,
                    nr,
                    unroll,
                    semiring,
                );
            }
        }
    }

    // A partial last line, unrolled at comptime.
    #[unroll]
    for unit in 0..tail {
        rank1_update::<E, EL, L, ER, V, RL>(
            lhs,
            lhs_scales,
            rhs,
            rhs_scales,
            c,
            &mut b,
            comptime!(lines * width + unit),
            comptime!(lines) as u32,
            0usize,
            comptime!(Some(unit)),
            contracted_per_step,
            along_k,
            columns,
            mr,
            nr,
            unroll,
            semiring,
        );
    }
}

/// One step `c += outer(A[:, k], B[k, :])` off the `k_line`-th line, each line under its scale.
/// `fixed` is the comptime component to extract; `None` takes `unit` at runtime.
///
/// Where a load holds `columns` runs of `contracted_per_step` values along the contraction, each
/// load is read once and split into its columns' runs, and each run's products with a row's line
/// go into that row's cell. A cell as wide as the run keeps one partial a lane, which the commit
/// sums. A one-wide cell, a block that lives across the walk, takes the run's sum each step, so
/// it holds one value however long a run is; under the sum of products the run's scale then
/// multiplies that sum, one product a cell rather than one a value, since a run lies inside one
/// scale.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, RL: Size>(
    lhs: &MatrixView<'_, Vector<EL, L>>,
    lhs_scales: &FactorReader,
    rhs: &MatrixView<'_, Vector<ER, RL>>,
    rhs_scales: &FactorReader,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    k: usize,
    k_line: u32,
    unit: usize,
    #[comptime] fixed: Option<usize>,
    #[comptime] contracted_per_step: usize,
    #[comptime] along_k: bool,
    #[comptime] columns: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] unroll: bool,
    #[comptime] semiring: Semiring,
) {
    if comptime!(along_k) {
        let cell = V::value();
        comptime!(assert!(
            cell == contracted_per_step || cell == 1,
            "mm: a cell holds a run's {contracted_per_step} partials or their sum, not {cell} values"
        ));
        let summed = comptime!(cell == 1);
        let scale_sum = comptime!(summed && semiring == Semiring::SUM_PROD);
        let mut a = Array::<Vector<E, L>>::new(mr);
        #[unroll(unroll)]
        for i in 0..mr {
            let pos = (i as u32, k_line);
            a[i] = lhs_scales.apply::<E, L>(Vector::<E, L>::cast_from(lhs.read(pos)), pos);
        }
        #[unroll(unroll)]
        for g in 0..comptime!(nr / columns) {
            let held = rhs.read((g as u32, k_line));
            #[unroll]
            for j in 0..columns {
                let n = g * columns + j;
                let pos = (n as u32, k_line);
                // In `E`, so an integer operand doesn't round the scale or wrap.
                let values = values_at::<E, ER, RL, L>(
                    held,
                    comptime!(j * contracted_per_step),
                    comptime!(Packing::Plain),
                );
                let values = if comptime!(scale_sum) {
                    values
                } else {
                    rhs_scales.apply::<E, L>(values, pos)
                };
                #[unroll(unroll)]
                for i in 0..mr {
                    let at = i * nr + n;
                    if comptime!(summed) {
                        let mut sum = Vector::<E, V>::cast_from(Monoid::identity::<E>(comptime!(
                            semiring.add()
                        )));
                        #[unroll]
                        for v in 0..contracted_per_step {
                            sum = semiring.step::<Vector<E, V>>(
                                Vector::<E, V>::cast_from(a[i].extract(v)),
                                Vector::<E, V>::cast_from(values.extract(v)),
                                sum,
                            );
                        }
                        if comptime!(scale_sum) {
                            sum = rhs_scales.apply::<E, V>(sum, pos);
                        }
                        c[at] = comptime!(semiring.add()).combine::<Vector<E, V>>(c[at], sum);
                    } else {
                        c[at] = semiring.step::<Vector<E, V>>(
                            Vector::<E, V>::cast_from(a[i]),
                            Vector::<E, V>::cast_from(values),
                            c[at],
                        );
                    }
                }
            }
        }
    } else {
        #[unroll(unroll)]
        for n in 0..nr {
            let pos = (k as u32, n as u32);
            b[n] = rhs_scales.apply::<E, V>(Vector::<E, V>::cast_from(rhs.read(pos)), pos);
        }
        #[unroll(unroll)]
        for i in 0..mr {
            let pos = (i as u32, k_line);
            let line = lhs_scales.apply::<E, L>(Vector::<E, L>::cast_from(lhs.read(pos)), pos);
            let a = if comptime!(fixed.is_some()) {
                Vector::<E, V>::cast_from(line.extract(comptime!(fixed.unwrap())))
            } else {
                Vector::<E, V>::cast_from(line.extract_dynamic(unit))
            };
            #[unroll(unroll)]
            for n in 0..nr {
                // A single fma; `+= a * b` would lower to a mul and a dependent add.
                c[i * nr + n] = semiring.step::<Vector<E, V>>(a, b[n], c[i * nr + n]);
            }
        }
    }
}

/// Checks shared by [`seed`] and [`commit`] before spreading a block column across sink cells.
fn assert_spread(contracted_per_step: usize, spread: usize, accumulator_width: usize, who: &str) {
    assert!(
        contracted_per_step == 1 || spread == 1,
        "{who}: units cannot hold contracted partials (contracted_per_step {contracted_per_step}) and neighbouring \
         sink cells (spread {spread}) at once"
    );
    assert!(
        spread == 1 || accumulator_width == 1,
        "{who}: a spread block scatters one unit per sink cell, so the accumulator must be \
         contracted_per_step scalar (it is {accumulator_width} wide)"
    );
}

/// Whether a spread block's units must be bounds-checked against the sink's `cols`.
pub(crate) fn spread_guard(spread: usize, cols: usize) -> bool {
    spread > 1 && !cols.is_multiple_of(spread)
}

/// Seed the `mr × nr` register block from the accumulator. Folded partials seed unit 0 and start
/// the rest at the identity; at `spread > 1` each block column gathers `spread` sink cells.
#[cube]
pub(crate) fn seed<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    #[comptime] contracted_per_step: usize,
    #[comptime] spread: usize,
    #[comptime] accumulator_width: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] unroll: bool,
) -> Array<Vector<E, V>> {
    comptime!(assert_spread(
        contracted_per_step,
        spread,
        accumulator_width,
        "block::seed"
    ));
    let guard = comptime!(spread_guard(spread, cols));
    let monoid = acc.monoid();
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            if comptime!(spread > 1) {
                let base = (n as u32).times(comptime!(spread as u32));
                let mut units = Vector::<E, V>::cast_from(Monoid::identity::<E>(monoid));
                #[unroll]
                for l in 0..spread {
                    let col = base.plus(comptime!(l as u32));
                    let live = if comptime!(guard) {
                        col < comptime!(cols as u32)
                    } else {
                        true.runtime()
                    };
                    if live {
                        units.insert(l, acc.seed((i as u32, col)).extract(0usize));
                    }
                }
                c[i * nr + n] = units;
            } else {
                let cell = acc.seed((i as u32, n as u32));
                if comptime!(contracted_per_step > 1) {
                    let mut units = Vector::<E, V>::cast_from(Monoid::identity::<E>(monoid));
                    units.insert(0usize, cell.extract(0usize));
                    c[i * nr + n] = units;
                } else {
                    c[i * nr + n] = Vector::<E, V>::cast_from(cell);
                }
            }
        }
    }
    c
}

/// Commit the block back through the accumulator, collapsing folded partials or scattering
/// spread units.
#[cube]
pub(crate) fn commit<E: Numeric, V: Size, A: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    c: Array<Vector<E, V>>,
    #[comptime] contracted_per_step: usize,
    #[comptime] spread: usize,
    #[comptime] accumulator_width: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] unroll: bool,
) {
    comptime!(assert_spread(
        contracted_per_step,
        spread,
        accumulator_width,
        "block::commit"
    ));
    let guard = comptime!(spread_guard(spread, cols));
    let unit_share = acc.unit_share();
    let monoid = acc.monoid();
    comptime!(assert!(
        !guard || !unit_share.reduces(),
        "block::commit: a spread block skips the units overhanging the sink, and a unit-split \
         accumulator ({unit_share:?}) folds across the plane on the way out, which that skip \
         would put under divergent control flow"
    ));
    #[unroll(unroll)]
    for i in 0..mr {
        #[unroll(unroll)]
        for n in 0..nr {
            let cell = c[i * nr + n];
            if comptime!(spread > 1) {
                let base = (n as u32).times(comptime!(spread as u32));
                #[unroll]
                for l in 0..spread {
                    let col = base.plus(comptime!(l as u32));
                    let live = if comptime!(guard) {
                        col < comptime!(cols as u32)
                    } else {
                        true.runtime()
                    };
                    if live {
                        acc.commit((i as u32, col), Vector::<E, A>::cast_from(cell.extract(l)));
                    }
                }
            } else if comptime!(contracted_per_step > 1) {
                let total = Monoid::reduce::<E, V>(cell, contracted_per_step, monoid);
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(total));
            } else {
                acc.commit((i as u32, n as u32), Vector::<E, A>::cast_from(cell));
            }
        }
    }
}
