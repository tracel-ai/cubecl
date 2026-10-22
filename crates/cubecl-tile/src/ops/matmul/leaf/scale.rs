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
    /// carrying the output's innermost axis.
    pub(crate) fn of(own: &Space, out: &Space) -> Side {
        match own.contains(out.axis_at(out.rank() - 1)) {
            true => Side::Rhs,
            false => Side::Lhs,
        }
    }
}

/// Refuse scales spanning axes only the other operand varies over.
pub(crate) fn check_scales_ride(side: Side, scales: &Space, output: &Space, axes: MatrixAxes) {
    let group = |range: core::ops::Range<usize>| {
        range
            .filter(|&p| scales.contains(output.axis_at(p)))
            .map(|p| output.axis_at(p))
            .collect::<Vec<_>>()
    };
    let rows = group(axes.row_split..axes.col_split);
    let cols = group(axes.col_split..output.rank());
    let (own, other, foreign) = match side {
        Side::Lhs => ("lhs", "rhs", cols),
        Side::Rhs => ("rhs", "lhs", rows),
    };
    assert!(
        foreign.is_empty(),
        "Tile::mul: the scales ride the {own} but span {foreign:?}, which only the {other} \
         varies over; a scale over both operands' own axes is a scale of the output, not a \
         factor of either term"
    );
}

/// Refuse a scales operand that reaches its block granularity by dividing an axis.
pub(crate) fn check_scales_omit_rather_than_divide(scales: &Projection) {
    for pa in 0..scales.physical_rank() {
        let divisor = scales.divisor(pa).bound();
        assert!(
            divisor == 1,
            "Tile::mul: this scales operand divides a logical axis by {divisor} to reach its \
             block. Spell the block as an axis of its own and omit the position inside it, so one \
             scale per block is what the operand's axes say rather than what its arithmetic does"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    fn contraction() -> (Space, MatrixAxes) {
        let out = Space::new(&[(M, 4), (N, 4)]);
        let lhs = Space::new(&[(M, 4), (K, 8)]);
        let axes = MatrixAxes::accumulator(&out, &lhs);
        (out, axes)
    }

    /// Row scales ride the lhs, column scales the rhs.
    #[test]
    fn scales_ride_the_operand_whose_axis_they_span() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(M, 4)]), &out, axes);
        check_scales_ride(Side::Rhs, &Space::new(&[(N, 4)]), &out, axes);
    }

    /// A scale over no matrix axis rides either side.
    #[test]
    fn a_scale_over_no_axis_rides_either_side() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(K, 8)]), &out, axes);
        check_scales_ride(Side::Rhs, &Space::new(&[(K, 8)]), &out, axes);
    }

    #[test]
    #[should_panic(expected = "ride the lhs but span [Axis(1)]")]
    fn column_scales_cannot_ride_the_lhs() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Lhs, &Space::new(&[(N, 4)]), &out, axes);
    }

    #[test]
    #[should_panic(expected = "ride the rhs but span [Axis(0)]")]
    fn row_scales_cannot_ride_the_rhs() {
        let (out, axes) = contraction();
        check_scales_ride(Side::Rhs, &Space::new(&[(M, 4)]), &out, axes);
    }
}
