//! The 2-D contraction nest: a single contracted axis, no gathered operands.

use cubecl::prelude::*;

use super::super::registers;
use super::super::scale::Side;
use super::shape::ContractShape;
use crate::*;

/// The 2-D contraction nest: over each batch matrix, the `mr × nr` accumulator block lives in
/// registers.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn contract<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] space: Space,
    #[comptime] contracted_per_step: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered,
        "contract: a gathered operand has no 2-D matrix view; it needs the N-D nest"
    ));

    let lw = lhs.vector_size();
    let rhs_load = rhs.vector_tile();
    let rw = comptime!(rhs_load.run_length());
    let aw = comptime!(acc.store.vector_size);
    comptime!(assert!(
        rw == aw || contracted_per_step > 1,
        "contract direct: a padded rhs staged wider than its {aw}-wide sink must use the N-D nest"
    ));
    let shape = comptime!(ContractShape::new(
        &lhs.place.space,
        &rhs.place.space,
        space,
        contracted_per_step,
        lw,
        rw,
        aw,
    ));
    comptime!(assert!(
        shape
            .matrix_axes(&lhs.place.space, &rhs.place.space)
            .is_some(),
        "contract: the 2-D nest reads each operand as one matrix, and no grouping of these axes \
         gives one; the N-D nest reads them a cell at a time"
    ));

    // A load holds this many runs along the contraction, each its own column.
    let columns = comptime!(rhs_load.values() / rw);
    if comptime!(contracted_per_step > 1) {
        let size!(W) = contracted_per_step;
        let size!(A) = 1usize;
        let size!(RL) = comptime!(rhs_load.values());
        nest::<E, EL, W, ER, W, A, RL>(acc, lhs, rhs, shape, columns, config, semiring);
    } else {
        let size!(W) = lw;
        let size!(A) = aw;
        nest::<E, EL, W, ER, A, A, A>(acc, lhs, rhs, shape, columns, config, semiring);
    }
}

/// The nest at fixed line widths: `L` the lhs's, `V` the rhs's and block's, `A` the accumulator's.
#[cube]
#[allow(clippy::too_many_arguments)]
fn nest<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size, RL: Size>(
    acc: &mut Memory,
    lhs: &Tile,
    rhs: &Tile,
    #[comptime] shape: ContractShape,
    #[comptime] columns: usize,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    let mr = comptime!(shape.mr);
    let nr = comptime!(shape.nr);
    let cols = comptime!(shape.cols);
    let kc = comptime!(shape.kc);
    let contracted_per_step = comptime!(shape.contracted_per_step);
    let lw = comptime!(shape.lw);
    let aw = comptime!(shape.aw);
    let matrices = comptime!(shape.matrices());

    let lhs_axes = comptime!(shape.lhs_axes(&lhs.place.space));
    let rhs_axes = comptime!(shape.rhs_axes(&rhs.place.space));

    let lhs_k_lines = comptime!(kc.div_ceil(lw));
    let component_fanout = comptime!(config.component_fanout);

    // Every value this nest reads is read under its factor, so an `e2m1` operand is read placed,
    // its lift carried by the factor.
    let (lhs, rhs) = (lhs.placed(), rhs.placed());
    for mat in 0..matrices {
        let lhs_mat = lhs.matrix_packed::<EL, L>(lhs_axes, mat);
        let rhs_mat = rhs.matrix_packed::<ER, RL>(rhs_axes, mat);
        let lhs_scales = lhs.reader(
            lhs_axes,
            mat,
            comptime!(Side::Lhs),
            comptime!(shape.space.clone()),
            comptime!(shape.acc_axes),
        );
        let rhs_scales = rhs.reader(
            rhs_axes,
            mat,
            comptime!(Side::Rhs),
            comptime!(shape.space.clone()),
            comptime!(shape.acc_axes),
        );
        let mut acc_view = acc.matrix_accumulate::<E, A>(
            mat,
            comptime!(shape.acc_axes),
            comptime!(shape.space.clone()),
            comptime!(semiring.add()),
        );

        // Split into an in-bounds body kept in registers and a checked edge body.
        let lhs_check = comptime!(lhs_mat.check);
        let rhs_check = comptime!(rhs_mat.check);
        let acc_check = acc_view.check();
        let eligible = comptime!(shape.scalars() <= config.budget);
        let split_edge =
            comptime!(eligible && config.split_edge && (lhs_check || rhs_check || acc_check));
        let in_bounds = if comptime!(split_edge) {
            let origin = (0u32.runtime(), 0u32.runtime());
            let lhs_extent = (
                comptime!(mr as u32).runtime(),
                comptime!(lhs_k_lines as u32).runtime(),
            );
            let rhs_extent = if comptime!(contracted_per_step > 1) {
                (
                    comptime!((nr / columns) as u32).runtime(),
                    comptime!((kc / contracted_per_step) as u32).runtime(),
                )
            } else {
                (
                    comptime!(kc as u32).runtime(),
                    comptime!(nr as u32).runtime(),
                )
            };
            let acc_extent = (
                comptime!(mr as u32).runtime(),
                comptime!(nr as u32).runtime(),
            );
            lhs_mat.block_in_bounds(origin, lhs_extent)
                && rhs_mat.block_in_bounds(origin, rhs_extent)
                && acc_view.block_in_bounds(origin, acc_extent)
        } else {
            false.runtime()
        };

        if comptime!(split_edge) {
            if in_bounds {
                body::<E, EL, L, ER, V, A, RL>(
                    &mut acc_view,
                    &lhs_mat,
                    &lhs_scales,
                    &rhs_mat,
                    &rhs_scales,
                    lw,
                    contracted_per_step,
                    columns,
                    aw,
                    mr,
                    nr,
                    cols,
                    kc,
                    true,
                    component_fanout,
                    semiring,
                );
            } else {
                body::<E, EL, L, ER, V, A, RL>(
                    &mut acc_view,
                    &lhs_mat,
                    &lhs_scales,
                    &rhs_mat,
                    &rhs_scales,
                    lw,
                    contracted_per_step,
                    columns,
                    aw,
                    mr,
                    nr,
                    cols,
                    kc,
                    false,
                    component_fanout,
                    semiring,
                );
            }
        } else {
            let unroll = comptime!(eligible && !lhs_check && !rhs_check && !acc_check);
            body::<E, EL, L, ER, V, A, RL>(
                &mut acc_view,
                &lhs_mat,
                &lhs_scales,
                &rhs_mat,
                &rhs_scales,
                lw,
                contracted_per_step,
                columns,
                aw,
                mr,
                nr,
                cols,
                kc,
                unroll,
                component_fanout,
                semiring,
            );
        }
    }
}

/// The nest body, either unrolled in registers or the checked edge fallback.
#[cube]
#[allow(clippy::too_many_arguments)]
fn body<E: Numeric, EL: Numeric, L: Size, ER: Numeric, V: Size, A: Size, RL: Size>(
    acc: &mut AccumulateView<'_, E, A>,
    lhs: &MatrixView<'_, Vector<EL, L>>,
    lhs_scales: &FactorReader,
    rhs: &MatrixView<'_, Vector<ER, RL>>,
    rhs_scales: &FactorReader,
    #[comptime] lw: usize,
    #[comptime] contracted_per_step: usize,
    #[comptime] columns: usize,
    #[comptime] aw: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] cols: usize,
    #[comptime] kc: usize,
    #[comptime] unroll: bool,
    #[comptime] component_fanout: bool,
    #[comptime] semiring: Semiring,
) {
    let mut c =
        registers::seed::<E, V, A>(acc, contracted_per_step, 1usize, aw, mr, nr, cols, unroll);
    registers::contract::<E, EL, L, ER, V, RL>(
        lhs,
        lhs_scales,
        rhs,
        rhs_scales,
        &mut c,
        lw,
        contracted_per_step,
        contracted_per_step > 1,
        columns,
        mr,
        nr,
        kc,
        unroll,
        component_fanout,
        semiring,
    );
    registers::commit::<E, V, A>(
        acc,
        c,
        contracted_per_step,
        1usize,
        aw,
        mr,
        nr,
        cols,
        unroll,
    );
}
