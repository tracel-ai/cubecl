//! N-D contraction dispatch for multiple contracted axes, projected operands and procedural
//! filters.

use cubecl::prelude::*;

use super::{nd, separable};

use super::super::shape::ContractShape;
use crate::*;

/// How the lhs varies over the accumulator's innermost axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum LhsRole {
    /// Free of the column: one read serves every cell of a row.
    FreeOfColumn,
    /// Lined along the column: the line it reads is the cell.
    LinedAlongColumn,
    /// Spans the column without lining along it: read per cell.
    PerCell,
}

/// How the rhs varies over the accumulator's row.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum RhsRole {
    /// Free of the row: its `nr` lines are reused down every row.
    FreeOfRow,
    /// Varies down the rows but holds for a whole row of cells.
    PerRow,
    /// Varies cell by cell.
    PerCell,
}

/// The accumulator scope at which one factor's complete tap walk is computed and cached.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FactorReuse {
    /// Once for the entire accumulator block.
    Block,
    /// Once for each accumulator row.
    Row,
    /// Once for each accumulator column.
    Column,
    /// Once for each accumulator cell.
    Cell,
}

impl FactorReuse {
    /// The scope a factor varying over these two axes can be cached at.
    pub(crate) fn of(varies_row: bool, varies_col: bool) -> Self {
        match (varies_row, varies_col) {
            (false, false) => FactorReuse::Block,
            (true, false) => FactorReuse::Row,
            (false, true) => FactorReuse::Column,
            (true, true) => FactorReuse::Cell,
        }
    }
}

/// The gather-specific comptime geometry over the shared [`ContractShape`].
#[derive(Clone, Debug)]
pub(crate) struct GatherProblem {
    pub block: ContractShape,
    pub lhs_space: Space,
    pub rhs_space: Space,
    /// The lhs's factor count, one per contracted axis; `None` takes the general schedule.
    pub factors: Option<usize>,
    /// Factor-local normalization requested by the procedural lhs.
    pub normalization: Option<Normalization>,
    /// The separable walk's weight count: taps of each factor, summed.
    pub taps: usize,
    /// Where each factor's taps start in that walk.
    pub offsets: Vec<usize>,
    /// Maximal reuse of each factor's tap walk within one accumulator block.
    pub factor_reuse: Vec<FactorReuse>,
    /// How each operand varies over the accumulator's own axes.
    pub lhs: LhsRole,
    pub rhs: RhsRole,
}

impl GatherProblem {
    #[allow(clippy::too_many_arguments)]
    fn new(
        lhs: &Space,
        rhs: &Space,
        rhs_projection: &Projection,
        block: ContractShape,
        factors: Option<usize>,
        factor_dependencies: Option<Vec<(bool, bool)>>,
        normalization: Option<Normalization>,
        rhs_boundaries: &[Option<Boundary>],
    ) -> Self {
        let rank = block.space.rank();
        let col = block.space.axis_at(rank - 1);
        // A separable lhs answers one scalar at a time, so it is never col-lined.
        let lhs_role = match lhs.contains(col) {
            false => LhsRole::FreeOfColumn,
            true if factors.is_none() && lhs.axis_at(lhs.rank() - 1) == col => {
                LhsRole::LinedAlongColumn
            }
            true => LhsRole::PerCell,
        };
        let rhs_role = match rhs.contains(block.space.axis_at(rank - 2)) {
            false => RhsRole::FreeOfRow,
            true if rhs.contains(col) => RhsRole::PerCell,
            true => RhsRole::PerRow,
        };
        assert_operand_shapes(
            lhs,
            rhs,
            &block.space,
            &block.reduce,
            block.lw,
            block.vw,
            lhs_role,
        );
        if let Some(factors) = factors {
            assert_eq!(
                factors,
                block.reduce.len(),
                "contract gather: a separable lhs needs one factor per contracted axis"
            );
            assert_separable_shapes(rhs_projection, &block.space, rhs.contains(col));
        }
        if normalization.is_some() {
            assert!(
                factors.is_some(),
                "contract gather: factor normalization needs a separable lhs"
            );
        }
        if let Some(normalization) = &normalization {
            for &axis in &block.reduce {
                assert!(
                    normalization.over.contains(axis)
                        && normalization.over.extent_raw(axis) == lhs.extent_raw(axis),
                    "contract gather: a normalized factor axis cannot be partitioned between \
                     .normalized() and the gather leaf; calling .normalized() below a split \
                     normalizes each chunk independently"
                );
            }
        }
        if matches!(&normalization, Some(n) if n.taps == TapSupport::InBounds) {
            assert_factorized_mask(rhs_projection, &block.reduce);
        }
        let row = block.space.axis_at(rank - 2);
        let factor_reuse = match (factors, factor_dependencies) {
            (Some(factors), Some(dependencies)) => {
                assert_eq!(dependencies.len(), factors);
                dependencies
                    .into_iter()
                    .enumerate()
                    .map(|(f, (mut varies_row, mut varies_col))| {
                        if matches!(&normalization, Some(n) if n.taps == TapSupport::InBounds) {
                            let tap = block.reduce[f];
                            varies_row |=
                                masked_bound_depends_on(rhs_projection, rhs_boundaries, tap, row);
                            varies_col |=
                                masked_bound_depends_on(rhs_projection, rhs_boundaries, tap, col);
                        }
                        FactorReuse::of(varies_row, varies_col)
                    })
                    .collect()
            }
            (None, None) => Vec::new(),
            _ => panic!("contract gather: factor dependencies must accompany the factorization"),
        };
        let offsets = block
            .reduce_extents
            .iter()
            .scan(0, |start, taps| {
                let at = *start;
                *start += taps;
                Some(at)
            })
            .collect();

        Self {
            taps: block.reduce_extents.iter().sum(),
            block,
            lhs_space: lhs.clone(),
            rhs_space: rhs.clone(),
            offsets,
            factor_reuse,
            factors,
            normalization,
            lhs: lhs_role,
            rhs: rhs_role,
        }
    }
}

/// Whether a rectangular source mask factorizes: no physical input axis is moved by two
/// contracted axes.
fn assert_factorized_mask(rhs: &Projection, reduce: &[Axis]) {
    for (f, &axis) in reduce.iter().enumerate() {
        if !rhs.logical_axes().contains(&axis) {
            continue;
        }
        for pa in rhs.carriers(axis) {
            for &other in &reduce[(f + 1)..] {
                assert!(
                    !rhs.physical_axis(pa)
                        .terms()
                        .iter()
                        .any(|term| term.axis == other),
                    "contract gather: TapSupport::InBounds needs each contracted axis to move distinct \
                     input axes; {axis:?} and {other:?} both move physical axis {pa}"
                );
            }
        }
    }
}

/// Whether the physical bound tested for `tap` also moves with accumulator `axis`.
fn masked_bound_depends_on(
    rhs: &Projection,
    boundaries: &[Option<Boundary>],
    tap: Axis,
    axis: Axis,
) -> bool {
    rhs.logical_axes().contains(&tap)
        && rhs.carriers(tap).iter().any(|&pa| {
            boundaries.get(pa).copied().flatten() == Some(Boundary::Zero)
                && rhs
                    .physical_axis(pa)
                    .terms()
                    .iter()
                    .any(|term| term.axis == axis)
        })
}

/// N-D variant of [`direct::contract`](super::direct::contract) for multiple contracted axes or
/// projected operands; a separable lhs takes its own schedule.
#[cube]
pub(crate) fn contract<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] space: Space,
    #[comptime] contracted_per_step: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lw = lhs.vector_size();
    let rw = rhs.vector_size();
    let aw = comptime!(acc.store.vector_size);
    // `step_served` refused anything but a matched pair or scalar sink, so this divides exactly.
    comptime!(assert!(
        rw == aw || aw == 1,
        "contract gather: a rhs staged wider than its sink spreads its units across scalar cells, \
         so the accumulator must be served scalar (rhs {rw}, accumulator {aw})"
    ));
    let factors = lhs.factors();
    let normalization = lhs.factor_normalization();
    let rhs_projection = rhs.projection();
    let rhs_boundaries = rhs.separable_boundaries();
    let rank = comptime!(space.rank());
    let factor_dependencies = lhs.factor_dependencies(
        factors,
        comptime!(space.axis_at(rank - 2)),
        comptime!(space.axis_at(rank - 1)),
    );

    let problem = comptime!(GatherProblem::new(
        &lhs.place.space,
        &rhs.place.space,
        &rhs_projection,
        ContractShape::new(
            &lhs.place.space,
            &rhs.place.space,
            space,
            contracted_per_step,
            lw,
            rw,
            aw
        ),
        factors,
        factor_dependencies,
        normalization,
        &rhs_boundaries,
    ));

    if comptime!(factors.is_some()) {
        // A separable lhs serves one value a step; the block is the accumulator's width.
        comptime!(assert!(
            contracted_per_step == 1 && lw == 1,
            "contract gather: a separable lhs needs scalar weights contracted_per_step one value a step"
        ));
        let size!(V) = rw;
        let size!(A) = aw;
        separable::contract::<E, EL, ER, V, A>(acc, lhs, rhs, problem, config, semiring);
    } else if comptime!(contracted_per_step > 1) {
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        nd::nest::<E, EL, W, ER, W, A>(acc, lhs, rhs, problem, config, semiring);
    } else {
        let size!(W) = lw;
        let size!(V) = rw;
        let size!(A) = aw;
        nd::nest::<E, EL, W, ER, V, A>(acc, lhs, rhs, problem, config, semiring);
    }
}

/// Assert the extra operand shapes the separable schedule assumes; fails at comptime.
fn assert_separable_shapes(rhs: &Projection, acc: &Space, rhs_spans_col: bool) {
    let col = acc.axis_at(acc.rank() - 1);
    assert!(
        rhs_spans_col,
        "contract gather: the separable schedule walks the accumulator's columns by stepping the \
         rhs's innermost physical axis, so the rhs must span {col:?}"
    );
    let innermost = rhs.physical_axis(rhs.physical_rank() - 1);
    assert!(
        innermost.is_identity(col),
        "contract gather: the separable schedule steps the rhs's innermost physical axis once per \
         accumulator column, so that axis must be {col:?} at coefficient 1"
    );
}

/// Assert each operand's vectorized axis is the one it lines along; fails at comptime.
#[allow(clippy::too_many_arguments)]
fn assert_operand_shapes(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    reduce: &[Axis],
    lhs_vec_len: usize,
    rhs_vec_len: usize,
    lhs_role: LhsRole,
) {
    assert!(
        !reduce.is_empty(),
        "contract gather: the operands contract no axis against the accumulator"
    );
    // An rhs-only contracted axis would take the `fastest` slot; name that constraint instead.
    for &axis in reduce {
        assert!(
            lhs.contains(axis),
            "contract gather: the lhs must span every contracted axis, but {axis:?} is contracted by \
             the rhs alone"
        );
    }
    let fastest = reduce[reduce.len() - 1];
    // A col-lined lhs lines along the accumulator, so every contracted axis is walked in elements.
    assert!(
        lhs_role == LhsRole::LinedAlongColumn || lhs.axis_at(lhs.rank() - 1) == fastest,
        "contract gather: the lhs must line along the fastest contracted axis {fastest:?}"
    );
    // A scalar rhs need not span either axis, so a weight shared by every column can omit it.
    let rhs_lined = rhs.axis_at(rhs.rank() - 1);
    assert!(
        rhs_vec_len == 1 || rhs_lined == acc.axis_at(acc.rank() - 1) || rhs_lined == fastest,
        "contract gather: a vectorized rhs must line along the accumulator's innermost axis or \
         the fastest contracted axis {fastest:?}"
    );
    // A per-cell lhs read covers `rhs_vec_len` columns only when lined along the column axis.
    assert!(
        lhs_role != LhsRole::PerCell || rhs_vec_len == 1,
        "contract gather: an lhs spanning the accumulator's innermost axis needs a value per \
         column, so that axis must be the one it lines along (the accumulator is {rhs_vec_len} \
         wide, the lhs lines {lhs_vec_len})"
    );
    assert!(
        lhs_role != LhsRole::LinedAlongColumn || lhs_vec_len == rhs_vec_len,
        "contract gather: a col-lined lhs is read as the cell itself, so its line width \
         ({lhs_vec_len}) must be the accumulator's ({rhs_vec_len})"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K0: Axis = Axis(2);
    const K1: Axis = Axis(3);

    fn spaces() -> (Space, Space, Space) {
        (
            Space::new(&[(M, 4), (K0, 2), (K1, 3)]),
            Space::new(&[(K0, 2), (K1, 3), (N, 8)]),
            Space::new(&[(M, 4), (N, 8)]),
        )
    }

    fn problem(factors: Option<usize>, vw: usize) -> GatherProblem {
        let (lhs, rhs, acc) = spaces();
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, vw, vw);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            factors,
            factors.map(|n| vec![(true, true); n]),
            None,
            &[],
        )
    }

    #[test]
    fn problem_derives_one_consistent_gather_geometry() {
        let problem = problem(Some(2), 4);
        let block = &problem.block;

        assert_eq!(block.reduce, vec![K0, K1]);
        assert_eq!(block.reduce_extents, vec![2, 3]);
        assert_eq!(problem.offsets, vec![0, 2]);
        assert_eq!((block.kc, problem.taps), (6, 5));
        assert_eq!((block.mr, block.nr), (4, 2));
        assert_eq!(block.batch_extents(), Vec::<usize>::new());
        assert_eq!(block.matrices(), 1);
        assert_eq!(block.scalars(), 32);
    }

    #[test]
    #[should_panic(expected = "one factor per contracted axis")]
    fn problem_rejects_a_factor_count_that_does_not_match_the_reduction() {
        problem(Some(3), 1);
    }

    /// A rank-one factorization of a two-axis reduction is rejected.
    #[test]
    #[should_panic(expected = "one factor per contracted axis")]
    fn problem_rejects_a_rank_one_factorization_of_a_two_axis_reduction() {
        problem(Some(1), 1);
    }

    #[test]
    #[should_panic(expected = "normalized factor axis cannot be partitioned")]
    fn problem_rejects_chunk_local_factor_normalization() {
        let (lhs, rhs, acc) = spaces();
        let original = Space::new(&[(M, 4), (K0, 4), (K1, 3)]);
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(true, true); 2]),
            Some(Normalization::new(
                TapSupport::Whole,
                DivGuard::default(),
                original,
            )),
            &[],
        );
    }

    #[test]
    #[should_panic(expected = "needs each contracted axis to move distinct input axes")]
    fn problem_rejects_masked_normalization_when_contracted_axes_share_input_axis() {
        let (lhs, rhs, acc) = spaces();
        let map = vec![
            PhysicalAxisMap::affine(&[(K0, 1), (K1, 1)]),
            PhysicalAxisMap::of(N),
        ];
        let projection = Projection::new(&[K0, K1, N], &map);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(true, true); 2]),
            Some(Normalization::new(
                TapSupport::InBounds,
                DivGuard::default(),
                lhs.clone(),
            )),
            &[],
        );
    }

    #[test]
    fn a_masked_bound_blocks_only_the_unsafe_column_hoist() {
        let (lhs, rhs, acc) = spaces();
        let map = vec![
            PhysicalAxisMap::affine(&[(K0, 1), (N, 1)]),
            PhysicalAxisMap::of(K1),
            PhysicalAxisMap::of(N),
        ];
        let projection = Projection::new(&[K0, K1, N], &map);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        let problem = GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(false, false); 2]),
            Some(Normalization::new(
                TapSupport::InBounds,
                DivGuard::default(),
                lhs.clone(),
            )),
            &[Some(Boundary::Zero), None, None],
        );

        assert_eq!(problem.factor_reuse[0], FactorReuse::Column);
        assert_eq!(problem.factor_reuse[1], FactorReuse::Block);
    }

    #[test]
    fn recipe_dependencies_choose_the_maximal_factor_cache() {
        let (lhs, rhs, acc) = spaces();
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        let problem = GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(true, false), (false, true)]),
            None,
            &[],
        );

        assert_eq!(
            problem.factor_reuse,
            vec![FactorReuse::Row, FactorReuse::Column]
        );
    }

    #[test]
    fn constant_and_fully_dependent_factors_choose_the_extreme_caches() {
        let (lhs, rhs, acc) = spaces();
        let projection = Projection::direct(&[K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, 1, 1, 1);
        let problem = GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            Some(2),
            Some(vec![(false, false), (true, true)]),
            None,
            &[],
        );

        assert_eq!(
            problem.factor_reuse,
            vec![FactorReuse::Block, FactorReuse::Cell]
        );
    }

    /// An unfactorized lhs leaves the reduction's rank unconstrained.
    #[test]
    fn problem_accepts_an_unfactorized_lhs() {
        assert_eq!(problem(None, 4).factors, None);
    }

    /// A batched problem whose lhs spans one axis with every operand, in the given order.
    fn batched(lhs: &[(Axis, usize)], factors: Option<usize>, width: usize) -> GatherProblem {
        let lhs = Space::new(lhs);
        let rhs = Space::new(&[(M, 4), (K0, 2), (K1, 3), (N, 8)]);
        let acc = Space::new(&[(M, 4), (N, 8)]);
        let projection = Projection::direct(&[M, K0, K1, N]);
        let block = ContractShape::new(&lhs, &rhs, acc, 1, width, width, width);
        GatherProblem::new(
            &lhs,
            &rhs,
            &projection,
            block,
            factors,
            factors.map(|n| vec![(true, true); n]),
            None,
            &[],
        )
    }

    /// A col-lined lhs is read as the cell.
    #[test]
    fn a_col_lined_lhs_is_read_as_the_cell() {
        let problem = batched(&[(K0, 2), (K1, 3), (N, 8)], None, 4);

        assert_eq!(problem.lhs, LhsRole::LinedAlongColumn);
        assert_eq!(problem.rhs, RhsRole::PerCell);
    }

    /// An lhs spanning the column off its line is read per cell.
    #[test]
    fn an_lhs_spanning_the_column_off_its_line_is_read_per_cell() {
        let problem = batched(&[(K0, 2), (N, 8), (K1, 3)], None, 1);

        assert_eq!(problem.lhs, LhsRole::PerCell);
    }

    /// A separable lhs is never col-lined.
    #[test]
    fn a_separable_lhs_is_never_col_lined() {
        let problem = batched(&[(K0, 2), (N, 8), (K1, 3)], Some(2), 1);

        assert_eq!(problem.lhs, LhsRole::PerCell);
    }
}
