//! Checks on a scales operand against the values it covers.

use crate::*;
/// Which factor of a contraction an operand is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Side {
    /// The left factor.
    Lhs,
    /// The right factor.
    Rhs,
}

impl Side {
    /// The side a factor spanning `own` takes in a contraction into `out`: the rhs is the one
    /// carrying the output's innermost dimension.
    pub(crate) fn of(own: &Space, out: &Space) -> Side {
        match own.contains(out.dimension_at(out.rank() - 1)) {
            true => Side::Rhs,
            false => Side::Lhs,
        }
    }
}

impl Side {
    /// Refuse a contraction into `out` whose factors, spanning `lhs` and `rhs`, are not on the
    /// sides their dimensions give them ([`Side::of`]): the left one carrying the output's rows,
    /// the right one its innermost dimension. An instruction that takes its operands by position
    /// would contract each against the other's dimension and drain wrong numbers rather than fail.
    pub(crate) fn refuse_misplaced(lhs: &Space, rhs: &Space, out: &Space) {
        for (factor, space, side) in [("left", lhs, Side::Lhs), ("right", rhs, Side::Rhs)] {
            assert!(
                Side::of(space, out) == side,
                "mma: the {factor} factor spans {:?}, which puts it on the other side of an \
                 output spanning {:?}: the right factor is the one carrying the output's \
                 innermost dimension. Hand the factors over in that order, or open the output in the \
                 order they are handed over in",
                space.dimensions().collect::<Vec<_>>(),
                out.dimensions().collect::<Vec<_>>()
            );
        }
    }
}

/// Refuse scales spanning dimensions only the other operand varies over.
pub(crate) fn check_scales_ride(
    side: Side,
    scales: &Space,
    output: &Space,
    dimensions: MatrixDimensions,
) {
    let group = |range: core::ops::Range<usize>| {
        range
            .filter(|&p| scales.contains(output.dimension_at(p)))
            .map(|p| output.dimension_at(p))
            .collect::<Vec<_>>()
    };
    let rows = group(dimensions.row_split..dimensions.col_split);
    let cols = group(dimensions.col_split..output.rank());
    let (own, other, foreign) = match side {
        Side::Lhs => ("lhs", "rhs", cols),
        Side::Rhs => ("rhs", "lhs", rows),
    };
    assert!(
        foreign.is_empty(),
        "Tile::mul: the scales ride the {own} but span {foreign:?}, which only the {other} \
         varies over; a scale over both operands' own dimensions is a scale of the output, not a \
         factor of either term"
    );
}

/// Refuse a scales operand that reaches its block granularity by dividing a dimension.
pub(crate) fn check_scales_omit_rather_than_divide(scales: &Projection) {
    for pa in 0..scales.physical_rank() {
        let divisor = scales.divisor(pa).bound();
        assert!(
            divisor == 1,
            "Tile::mul: this scales operand divides a dimension by {divisor} to reach its \
             block. Spell the block as a dimension of its own and omit the position inside it, so one \
             scale per block is what the operand's dimensions say rather than what its arithmetic does"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);
    const K: Dimension = Dimension(2);

    fn contraction() -> (Space, MatrixDimensions) {
        let out = Space::new(&[(M, 4), (N, 4)]);
        let lhs = Space::new(&[(M, 4), (K, 8)]);
        let dimensions = MatrixDimensions::accumulator(&out, &lhs);
        (out, dimensions)
    }

    /// Row scales ride the lhs, column scales the rhs.
    #[test]
    fn scales_ride_the_operand_whose_dimension_they_span() {
        let (out, dimensions) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(M, 4)]), &out, dimensions);
        check_scales_ride(Side::Rhs, &Space::new(&[(N, 4)]), &out, dimensions);
    }

    /// A scale over no matrix dimension rides either side.
    #[test]
    fn a_scale_over_no_dimension_rides_either_side() {
        let (out, dimensions) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(K, 8)]), &out, dimensions);
        check_scales_ride(Side::Rhs, &Space::new(&[(K, 8)]), &out, dimensions);
    }

    #[test]
    #[should_panic(expected = "ride the lhs but span [Dimension(1)]")]
    fn column_scales_cannot_ride_the_lhs() {
        let (out, dimensions) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(N, 4)]), &out, dimensions);
    }

    #[test]
    #[should_panic(expected = "ride the rhs but span [Dimension(0)]")]
    fn row_scales_cannot_ride_the_rhs() {
        let (out, dimensions) = contraction();
        check_scales_ride(Side::Rhs, &Space::new(&[(M, 4)]), &out, dimensions);
    }

    /// The factors of `c[m, n] += a[m, k] · b[k, n]` are on their sides, whichever order each
    /// lists its own dimensions in.
    #[test]
    fn factors_on_their_sides_are_admitted() {
        let out = Space::new(&[(M, 4), (N, 4)]);
        Side::refuse_misplaced(
            &Space::new(&[(M, 4), (K, 8)]),
            &Space::new(&[(K, 8), (N, 4)]),
            &out,
        );
        Side::refuse_misplaced(
            &Space::new(&[(K, 8), (M, 4)]),
            &Space::new(&[(N, 4), (K, 8)]),
            &out,
        );
    }

    /// Factors handed over the other way round would each contract against the other's dimension.
    #[test]
    #[should_panic(expected = "the left factor spans [Dimension(1), Dimension(2)]")]
    fn factors_handed_over_reversed_are_refused() {
        Side::refuse_misplaced(
            &Space::new(&[(N, 4), (K, 8)]),
            &Space::new(&[(M, 4), (K, 8)]),
            &Space::new(&[(M, 4), (N, 4)]),
        );
    }

    /// An output opened `(n, m)` puts the factors in the order `(rhs, lhs)`.
    #[test]
    #[should_panic(expected = "the left factor spans [Dimension(0), Dimension(2)]")]
    fn an_output_opened_the_other_way_round_is_refused() {
        Side::refuse_misplaced(
            &Space::new(&[(M, 4), (K, 8)]),
            &Space::new(&[(K, 8), (N, 4)]),
            &Space::new(&[(N, 4), (M, 4)]),
        );
    }
}
