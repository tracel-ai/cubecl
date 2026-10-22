//! The softmax leaf, composed from the row ops.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// One online-softmax fold step on this final score tile, in place.
    /// Returns the accumulator rescale `exp(m_old - m_new)` per owned row (1 if unowned).
    pub fn softmax<EA: Float>(
        &mut self,
        p: &mut Tile,
        state: &mut RowState<EA>,
        probe: &MaskProbe,
        mask: &Tile,
        scale: EA,
    ) -> Array<EA> {
        let define!(Out) = p.element;
        let corr = self.softmax_in_place::<EA>(state, probe, mask, scale);
        match comptime!(state.share) {
            RowShare::Unit { rows: _ } => self.write_rows_to::<EA, Out>(p, &*state),
            RowShare::Plane { rows, units } => self.write_rows_to_planar::<EA, Out>(p, rows, units),
        }
        corr
    }

    /// [`softmax`](Tile::softmax) leaving this step's unnormalized P in place of the scores.
    pub fn softmax_in_place<EA: Float>(
        &mut self,
        state: &mut RowState<EA>,
        probe: &MaskProbe,
        mask: &Tile,
        scale: EA,
    ) -> Array<EA> {
        let rank = comptime!(self.place.space.rank());
        // Rank, not finality: a score tile's instruction is a level; every read here is flat.
        comptime!(assert!(
            rank == 2,
            "softmax: a leaf op on rank-2 score tiles"
        ));
        comptime!(assert!(
            state.space.contains(self.place.space.axis_at(0))
                && !state.space.contains(self.place.space.axis_at(1)),
            "softmax reduces the score axis absent from the state's space; \
             v1 requires it to be the trailing axis"
        ));
        // The passes read a row a line at a time, so the lines must not straddle rows.
        let w = self.vector_size();
        comptime!(assert!(
            self.place.space.extent_at(1).is_multiple_of(w),
            "softmax: the score's line width divides its columns"
        ));

        let rows = comptime!(state.share.rows());
        let mut max_buf = Array::<EA>::new(rows);
        let mut sum_buf = Array::<EA>::new(rows);

        match comptime!(state.share) {
            RowShare::Unit { rows: _ } => {
                self.scale_and_mask::<EA>(scale, probe, mask, &*state);
                self.row_max::<EA>(&mut max_buf, &*state);
                self.exp_diff::<EA>(&max_buf, &*state);
                self.row_sum::<EA>(&mut sum_buf, &*state);
            }
            RowShare::Plane { rows, units } => {
                self.scale_and_mask_planar::<EA>(scale, probe, mask, rows, units);
                self.row_max_planar::<EA>(&mut max_buf, &state.m, rows, units);
                self.exp_diff_planar::<EA>(&max_buf, rows, units);
                self.row_sum_planar::<EA>(&mut sum_buf, rows, units);
            }
        }

        state.update(&max_buf, &sum_buf)
    }
}
