//! The contraction nest's entry point: routes to the 2-D or the N-D nest.

use cubecl::prelude::*;

use super::direct;
use super::gather;
use super::shape::ContractShape;
use crate::*;

/// Run the register instruction over each batch matrix. A scaled factor requires the 2-D nest.
#[cube]
pub(crate) fn contract<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_scaled = lhs.quantized();
    let rhs_scaled = rhs.quantized();
    let scaled = comptime!(lhs_scaled || rhs_scaled);

    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let lhs_procedural = lhs.is_procedural();
    let rhs_procedural = rhs.is_procedural();
    let lw = lhs.vector_size();
    // One column's run of a rhs load: the whole load unless it is stored across columns.
    let rhs_load = rhs.vector_tile();
    let rw = comptime!(rhs_load.run_length());
    let aw = comptime!(acc.store.vector_size);
    // One value at a time along the contraction, the rhs reads as no matrix: the N-D nest
    // addresses it at the cell.
    let rhs_along_k = comptime!(
        lhs.place
            .space
            .contains(rhs.place.space.axis_at(rhs.place.space.rank() - 1))
    );
    let contracted_per_step = comptime!(contracted_per_step(
        &lhs.place.space,
        &rhs.place.space,
        &space,
        lw,
        rw,
        aw
    ));
    // Several contracted axes still form one `k` edge when an operand carries them as one run.
    let shape = comptime!(ContractShape::new(
        &lhs.place.space,
        &rhs.place.space,
        space.clone(),
        contracted_per_step,
        lw,
        rw,
        aw
    ));
    let flat = comptime!(
        shape
            .matrix_axes(&lhs.place.space, &rhs.place.space)
            .is_some()
    );
    let nd = comptime!(
        !flat
            || lhs_gathered
            || rhs_gathered
            || lhs_procedural
            || rhs_procedural
            || (contracted_per_step == 1 && rw != aw)
            || (contracted_per_step == 1 && rhs_along_k)
    );
    comptime!(assert!(
        !(nd && scaled),
        "mm: a scaled factor reads as one matrix. This contraction needs the N-D nest, which \
         addresses every operand at the cell instead"
    ));

    if nd {
        gather::contract::<E, EL, ER>(acc, lhs, rhs, space, contracted_per_step, config, semiring);
    } else {
        direct::contract::<E, EL, ER>(acc, lhs, rhs, space, contracted_per_step, config, semiring);
    }
}

/// How many contracted values one step consumes, reconciled across both operands and the
/// accumulator.
pub(crate) fn contracted_per_step(
    lhs: &Space,
    rhs: &Space,
    acc: &Space,
    lw: usize,
    rw: usize,
    aw: usize,
) -> usize {
    let contracted = Space::contracted(&[lhs, rhs], acc);
    let k = contracted[contracted.len() - 1];
    let lined = rhs.axis_at(rhs.rank() - 1);
    if lined != k {
        assert!(
            rw == aw || aw == 1,
            "contract: the rhs lines along {lined:?} together with the accumulator, so both must \
             be served at one width unless the accumulator is scalar and the rhs is a padded \
             stage (rhs {rw}, accumulator {aw})"
        );
        return 1;
    }
    // A one-wide line has no direction: it is a step of one contracted value.
    let contracted_per_step = rhs.contracted_per_step(&contracted, rw);
    assert!(
        contracted_per_step > 1 || rw == 1,
        "contract: the rhs lines along the contracted axis {k:?}, which is served in whole lines; \
         its width {rw} must divide the axis's extent {:?}",
        rhs.extent_raw(k)
    );
    assert_eq!(
        lhs.contracted_per_step(&contracted, lw),
        contracted_per_step,
        "contract: the rhs serves {contracted_per_step} contracted values a step; line the lhs along {k:?} at \
         the same width (it is {lw} wide)"
    );
    assert_eq!(
        aw, 1,
        "contract: a step serving {contracted_per_step} contracted values holds partials of one cell in the \
         block's units, so the accumulator cannot also be served in {aw}-wide lines"
    );
    contracted_per_step
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    fn spaces(lhs: &[Axis], rhs: &[Axis]) -> (Space, Space, Space) {
        let extents = [(M, 4usize), (N, 4), (K, 8)];
        let pick = |axes: &[Axis]| {
            Space::new(
                &axes
                    .iter()
                    .map(|&a| (a, extents.iter().find(|e| e.0 == a).unwrap().1))
                    .collect::<Vec<_>>(),
            )
        };
        (pick(lhs), pick(rhs), pick(&[M, N]))
    }

    /// An rhs lined along the accumulator serves one value a step.
    #[test]
    fn an_rhs_lined_along_the_accumulator_serves_one_value_a_step() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[K, N]);
        assert_eq!(contracted_per_step(&lhs, &rhs, &acc, 4, 2, 2), 1);
    }

    /// Both operands lined along the contracted axis serve a line.
    #[test]
    fn both_operands_lined_along_the_contracted_axis_serve_a_line() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        assert_eq!(contracted_per_step(&lhs, &rhs, &acc, 4, 4, 1), 4);
    }

    /// A dynamic contracted axis serves a line.
    #[test]
    fn a_dynamic_contracted_axis_serves_a_line() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        let (lhs, rhs) = (lhs.with_dynamic(&[K]), rhs.with_dynamic(&[K]));
        assert_eq!(contracted_per_step(&lhs, &rhs, &acc, 4, 4, 1), 4);
    }

    /// A width that does not divide the contracted extent is refused.
    #[test]
    #[should_panic(expected = "served in whole lines")]
    fn a_width_that_misdivides_the_contracted_axis_is_refused() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        contracted_per_step(&lhs, &rhs, &acc, 3, 3, 1);
    }

    /// A folded step needs both operands lined.
    #[test]
    #[should_panic(expected = "line the lhs along")]
    fn a_folded_step_needs_both_operands_lined() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        contracted_per_step(&lhs, &rhs, &acc, 1, 4, 1);
    }

    /// A folded step needs a scalar accumulator.
    #[test]
    #[should_panic(expected = "cannot also be served")]
    fn a_folded_step_needs_a_scalar_accumulator() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[N, K]);
        contracted_per_step(&lhs, &rhs, &acc, 4, 4, 2);
    }

    /// The rhs shares the accumulator's width.
    #[test]
    #[should_panic(expected = "served at one width")]
    fn an_rhs_lined_along_the_accumulator_shares_its_width() {
        let (lhs, rhs, acc) = spaces(&[M, K], &[K, N]);
        contracted_per_step(&lhs, &rhs, &acc, 4, 1, 2);
    }
}
