//! The general N-D gather nest, K-major with each operand read hoisted to its widest reuse scope.

use cubecl::prelude::*;
use cubecl_std::tensor::layout::CoordsDyn;

use super::super::super::registers;
use crate::*;

use super::base::{GatherProblem, LhsRole, LhsVectorAxis, RhsRole};
use super::coords::{cell_read, step_coords};
use super::row_vectors::RowVectors;
use crate::ops::matmul::leaf::Side;

/// The nest at fixed vector widths: `L` the lhs's, `V` the rhs's and block's, `A` the
/// accumulator's.
#[cube]
pub(super) fn nest<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] problem: GatherProblem,
    #[comptime] config: SoftwareMatmul,
    #[comptime] semiring: Semiring,
) {
    let matrices = problem.block.matrices();
    let batch_extents = problem.block.batch_extents();

    let lhs_view = lhs.nd_packed::<EL, L>(comptime!(Guard::Checked));
    let rhs_view = rhs.nd_packed::<ER, V>(comptime!(Guard::Checked));
    // `comptime!`-bound so `#[unroll(flag)]` sees a comptime flag; otherwise it silently rolls.
    let lhs_check = comptime!(lhs_view.check);
    let rhs_check = comptime!(rhs_view.check);

    let eligible = problem.block.scalars() <= config.budget;
    // A clamp guard can't be retired by the corner check; see `Tile::guard_provable`.
    let lhs_provable = lhs.guard_provable();
    let rhs_provable = rhs.guard_provable();
    let provable = comptime!(lhs_provable && rhs_provable);
    // A spread block's last column reads past the proven box, so keep the leaf checked.
    let spread_overhang = comptime!(registers::spread_guard(
        problem.block.spread,
        problem.block.cols
    ));
    let split_operands =
        config.split_edge && eligible && provable && !spread_overhang && (lhs_check || rhs_check);
    let operands_inside = if split_operands {
        box_in_bounds::<EL, L>(&lhs_view, &problem.lhs_space, problem.block.lw)
            && box_in_bounds::<ER, V>(&rhs_view, &problem.rhs_space, problem.block.vw)
    } else {
        comptime!(false).runtime()
    };

    for mat in 0..matrices {
        let batch = Coords::constant(batch_extents.clone()).unravel(mat.cast::<u32>());

        let mut acc = acc.matrix_accumulate::<E, A>(
            mat,
            problem.block.acc_axes,
            &problem.block.space,
            semiring.add(),
        );

        let acc_check = acc.check();
        let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);

        // Interior instances prove the operand box once and read unguarded. The accumulator stays
        // guarded on both sides: it must never write outside the output.
        if split_operands {
            // `split_operands` implies `eligible`, and a proved walk reads no operand check, so
            // only the checked fallback stays rolled.
            if operands_inside {
                walk::<E, EL, L, ER, V, A>(
                    lhs,
                    &lhs.nd_packed::<EL, L>(comptime!(Guard::Proved)),
                    &rhs.nd_packed::<ER, V>(comptime!(Guard::Proved)),
                    &mut acc,
                    &batch,
                    problem.clone(),
                    config,
                    comptime!(true),
                    semiring,
                );
            } else {
                walk::<E, EL, L, ER, V, A>(
                    lhs,
                    &lhs_view,
                    &rhs_view,
                    &mut acc,
                    &batch,
                    problem.clone(),
                    config,
                    comptime!(false),
                    semiring,
                );
            }
        } else {
            walk::<E, EL, L, ER, V, A>(
                lhs,
                &lhs_view,
                &rhs_view,
                &mut acc,
                &batch,
                problem.clone(),
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
    #[comptime] space: &Space,
    #[comptime] width: usize,
) -> bool {
    let rank = space.rank();
    let extents = comptime!(crate::line_extents(space, width, 0, rank));

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
fn walk<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size>(
    lhs: &Tile,
    lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &Masked<'_, Vector<ER, V>, CoordsDyn>,
    acc: &mut AccumulateView<'_, E, A>,
    batch: &Coords<u32>,
    #[comptime] problem: GatherProblem,
    #[comptime] config: SoftwareMatmul,
    #[comptime] unroll: bool,
    #[comptime] semiring: Semiring,
) {
    // Comptime locals: `#[unroll(flag)]` and `0..n` bounds silently roll otherwise.
    let mr = problem.block.mr;
    let nr = problem.block.nr;
    let cols = problem.block.cols;
    let contracted_per_step = problem.block.contracted_per_step;
    let spread = problem.block.spread;
    let aw = problem.block.aw;
    let kc = problem.block.kc;

    let groups = comptime!(StepGroups::of(&problem, config));
    let source = comptime!(ComponentSource::new(&problem));

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

    // One rhs vector per column, reused by every row; allocated once for the whole K walk.
    let mut b = Array::<Vector<E, V>>::new(comptime!(nr));
    let mut row_vectors = RowVectors::<EL, L>::new(lhs, &problem);

    // Tap by tap over the contracted axes before the fastest, so only the fastest one moves in the
    // step loop.
    let extents = problem.block.reduce_extents.clone();
    let tap_extents = comptime!(extents[..extents.len() - 1].to_vec());
    let run = problem.block.fastest_extent();
    let taps = tap_extents.iter().product::<usize>();
    comptime!(assert!(
        run.is_multiple_of(contracted_per_step),
        "contract gather: a step contracts {contracted_per_step} values of the fastest contracted \
         axis, whose extent is {run}"
    ));
    // Scalars of the fastest axis one read covers, and the steps it serves.
    let group = groups.steps_per_read(&problem) * contracted_per_step;
    let full_groups = run / group;
    let tail = run % group / contracted_per_step;
    // A small register block can still contract a large filter (e.g. a depthwise weight
    // gradient). Bound tap expansion separately while keeping the cells in registers.
    // See https://github.com/tracel-ai/burn/issues/6054.
    let unroll_taps = unroll && taps <= config.budget;
    let unroll_steps = unroll && kc <= config.budget && contracted_per_step == 1;

    #[unroll(unroll_taps)]
    for tap in 0..taps {
        let tap_coords = Coords::constant(tap_extents.clone()).unravel(tap.cast::<u32>());
        row_vectors.anchor(lhs_view, batch, &tap_coords, problem.clone(), unroll);
        #[unroll(unroll_steps)]
        for read in 0..full_groups {
            read_steps::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut row_vectors,
                &mut c,
                &mut b,
                batch,
                &tap_coords,
                read * group,
                groups.steps_per_read(&problem),
                comptime!(groups),
                comptime!(source),
                unroll,
                problem.clone(),
                semiring,
            );
        }
        if tail > 0 {
            read_steps::<E, EL, L, ER, V>(
                lhs_view,
                rhs_view,
                &mut row_vectors,
                &mut c,
                &mut b,
                batch,
                &tap_coords,
                full_groups * group,
                comptime!(tail),
                comptime!(groups),
                comptime!(source),
                unroll,
                problem.clone(),
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

/// `steps` consecutive contracted steps of the tap `tap_coords` from scalar `first` of the fastest
/// axis, served by one read of the row vectors.
#[cube]
fn read_steps<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &Masked<'_, Vector<ER, V>, CoordsDyn>,
    row_vectors: &mut RowVectors<'_, EL, L>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    batch: &Coords<u32>,
    tap_coords: &Coords<u32>,
    first: usize,
    #[comptime] steps: usize,
    #[comptime] groups: StepGroups,
    #[comptime] source: ComponentSource,
    #[comptime] unroll: bool,
    #[comptime] problem: GatherProblem,
    #[comptime] semiring: Semiring,
) {
    let contracted_per_step = problem.block.contracted_per_step;
    row_vectors.read(
        lhs_view,
        batch,
        &step_coords(tap_coords, first.cast::<u32>()),
        problem.clone(),
        unroll,
    );
    #[unroll]
    for step in 0..steps {
        let reduce_coords = step_coords(
            tap_coords,
            (first + step * comptime!(contracted_per_step)).cast::<u32>(),
        );
        rank1_update::<E, EL, L, ER, V>(
            lhs_view,
            rhs_view,
            row_vectors,
            c,
            b,
            batch,
            &reduce_coords,
            groups.component(step, source),
            unroll,
            problem.clone(),
            semiring,
        );
    }
}

/// One gathered rank-1 update over the row vectors already read, each other read hoisted to the
/// coarsest cell its operand is invariant over.
#[cube]
fn rank1_update<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size>(
    lhs_view: &Masked<'_, Vector<EL, L>, CoordsDyn>,
    rhs_view: &Masked<'_, Vector<ER, V>, CoordsDyn>,
    row_vectors: &mut RowVectors<'_, EL, L>,
    c: &mut Array<Vector<E, V>>,
    b: &mut Array<Vector<E, V>>,
    batch: &Coords<u32>,
    reduce_coords: &Coords<u32>,
    #[comptime] source: ComponentSource,
    #[comptime] unroll: bool,
    #[comptime] problem: GatherProblem,
    #[comptime] semiring: Semiring,
) {
    let nr = problem.block.nr;
    let lw = problem.block.lw;
    let rows_per_vector = problem.rows_per_vector();

    let k_axis_idx = problem.block.reduce.len() - 1;
    if comptime!(problem.rhs == RhsRole::FreeOfRow) {
        #[unroll(unroll)]
        for n in 0..nr {
            b[n] = Vector::<E, V>::cast_from(cell_read::<ER, V>(
                rhs_view,
                batch,
                0u32,
                n as u32,
                reduce_coords,
                comptime!(Side::Rhs),
                problem.clone(),
            ));
        }
    }
    #[unroll(unroll)]
    for row_vector in 0..problem.row_vector_count() {
        #[unroll]
        for component in 0..rows_per_vector {
            let i = row_vector * rows_per_vector + component;
            // Row-invariant reads, taken once; zero and evaluated away when the cell loop reads
            // them.
            let mut a_row = Vector::<E, V>::cast_from(E::from_int(0));
            if comptime!(problem.lhs == LhsRole::FreeOfColumn) {
                a_row = vector_component::<E, EL, L, V>(
                    row_vectors.vector(row_vector),
                    reduce_coords,
                    source,
                    component,
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
                    reduce_coords,
                    comptime!(Side::Rhs),
                    problem.clone(),
                ));
            }
            #[unroll(unroll)]
            for n in 0..nr {
                let a = if comptime!(problem.lhs == LhsRole::PerCell) {
                    // Each column is a different value.
                    let vector = cell_read::<EL, L>(
                        lhs_view,
                        batch,
                        i as u32,
                        n as u32,
                        reduce_coords,
                        comptime!(Side::Lhs),
                        problem.clone(),
                    );
                    vector_component::<E, EL, L, V>(
                        vector,
                        reduce_coords,
                        source,
                        component,
                        lw,
                        k_axis_idx,
                    )
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
                        reduce_coords,
                        comptime!(Side::Rhs),
                        problem.clone(),
                    ))
                };
                // One semiring step, for the reason [`registers::rank1_update`] gives.
                c[i * nr + n] = semiring.step::<Vector<E, V>>(a, v, c[i * nr + n]);
            }
        }
    }
}

/// How the steps of one tap group under a read of the lhs, decided once from the problem.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StepGroups {
    /// One lhs vector serves `lw` consecutive steps, each its component at a fixed position: the
    /// vector is read once and every step takes its own component.
    PerVector,
    /// One read a step, taking the component [`ComponentSource`] names: scalar widths, several
    /// contracted values a step, vectors along the rows or columns, a component index that drifts
    /// from the walk, and an lhs spanning the columns without `component_fanout`.
    PerStep,
}

impl StepGroups {
    fn of(problem: &GatherProblem, config: SoftwareMatmul) -> Self {
        let fixed_components = problem.block.lw > 1
            && problem.block.contracted_per_step == 1
            && problem.lhs_vector_axis == LhsVectorAxis::Contracted
            && problem.block.component_index_exact()
            && (problem.lhs == LhsRole::FreeOfColumn || config.component_fanout);
        match fixed_components {
            true => Self::PerVector,
            false => Self::PerStep,
        }
    }

    /// The steps one read of the row vectors serves.
    fn steps_per_read(self, problem: &GatherProblem) -> usize {
        match self {
            Self::PerVector => problem.block.lw,
            Self::PerStep => 1,
        }
    }

    /// Where step `step` of a read takes its component from.
    fn component(self, step: usize, source: ComponentSource) -> ComponentSource {
        match self {
            Self::PerVector => ComponentSource::Fixed(step),
            Self::PerStep => source,
        }
    }
}

/// Which part of one lhs vector a contracted step reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ComponentSource {
    /// The whole vector: its components are the step's several contracted values, or the cells of
    /// a column-vectorized lhs.
    Whole,
    /// The component of the row being updated: each component of a row-vectorized lhs is a row.
    RowComponent,
    /// The component at this fixed position.
    Fixed(usize),
    /// The component the fastest contracted coordinate lands on, taken at run time.
    FromCoords,
}

impl ComponentSource {
    fn new(problem: &GatherProblem) -> Self {
        let sums_steps = problem.block.contracted_per_step > 1;
        match (problem.lhs_vector_axis, sums_steps, problem.block.lw) {
            (LhsVectorAxis::Columns, _, _) | (LhsVectorAxis::Contracted, true, _) => Self::Whole,
            (LhsVectorAxis::Rows, _, _) => Self::RowComponent,
            (LhsVectorAxis::Contracted, false, 1) => Self::Fixed(0),
            (LhsVectorAxis::Contracted, false, _) => Self::FromCoords,
        }
    }
}

/// The component of one lhs vector a step reads, widened into the accumulate
/// element. `component` is the row of a row-vectorized lhs.
#[cube]
fn vector_component<E: Numeric, EL: Numeric, L: Size, V: Size>(
    vector: Vector<EL, L>,
    reduce_coords: &Coords<u32>,
    #[comptime] source: ComponentSource,
    #[comptime] component: usize,
    #[comptime] lw: usize,
    #[comptime] k_axis_idx: usize,
) -> Vector<E, V> {
    match source {
        ComponentSource::Whole => Vector::<E, V>::cast_from(vector),
        ComponentSource::RowComponent => {
            Vector::<E, V>::cast_from(vector.extract(comptime!(component)))
        }
        ComponentSource::Fixed(position) => {
            Vector::<E, V>::cast_from(vector.extract(comptime!(position)))
        }
        ComponentSource::FromCoords => {
            let last_k = reduce_coords.at(comptime!(k_axis_idx));
            Vector::<E, V>::cast_from(
                vector.extract_dynamic((last_k % comptime!(lw as u32)).cast::<usize>()),
            )
        }
    }
}
