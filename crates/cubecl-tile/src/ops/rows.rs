//! Row verbs on a tile: publish per-row values, scale rows, copy owned rows.
//! No op here syncs; the caller owns the barriers.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// Publish per-owned-row `values` into this factors tile, one cell per score row.
    /// The caller syncs before any cross-worker read.
    pub fn store_rows<EA: Float>(&mut self, values: &Array<EA>, state: &RowState<EA>) {
        let share = comptime!(state.share);
        let rpu = comptime!(share.rows());
        let rows = comptime!(self.place.space.cells());
        comptime!(assert!(
            (0..self.place.space.rank())
                .filter(|&p| self.place.space.extent_at(p) > 1)
                .count()
                <= 1,
            "store_rows: a row unit, one cell per score row; this tile spans {:?}",
            self.place.space
        ));
        let size!(W) = self.vector_size();
        let mut view = self.flat_mut::<EA, W>();

        // One writer per row: the owning unit, or the owning plane's first unit.
        let writer = owned_unit(share) == 0;
        #[unroll]
        for ri in 0..rpu {
            let r = state.owned_row(ri);
            if r < rows && writer {
                view.write(r, Vector::cast_from(values[ri]));
            }
        }
    }

    /// Online-softmax rescale by the row's owner: `self[r, :] *= corr[ri]` over `share`'s rows.
    pub fn rescale_rows<EA: Float>(&mut self, corr: &Array<EA>, state: &RowState<EA>) {
        let share = comptime!(state.share);
        match &self.kind {
            TileKind::Memory(_) => self.rescale_rows_in_memory(corr, state),
            TileKind::PlanePartition(p) => {
                comptime!(match share {
                    RowShare::Plane { .. } => {}
                    RowShare::Unit { rows: _ } => {
                        panic!("rescale_rows: a plane-resident accumulator is owned by its plane")
                    }
                });
                p.rescale_rows(corr)
            }
            TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("rescale_rows: a memory tile or a plane-resident accumulator")
            }
        }
    }

    fn rescale_rows_in_memory<EA: Float>(&mut self, corr: &Array<EA>, state: &RowState<EA>) {
        let share = comptime!(state.share);
        let rank = comptime!(self.place.space.rank());
        let rows = comptime!(self.place.space.extent_at(rank - 2));
        let cols = comptime!(self.place.space.extent_at(rank - 1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let rpw = comptime!(share.rows());
        let units = comptime!(share.units());
        let plane_unit = owned_unit(share);
        let mut view = self.flat_mut::<EA, W>();
        #[unroll]
        for ri in 0..rpw {
            let r = state.owned_row(ri);
            if r < rows {
                let factor = Vector::<EA, W>::cast_from(corr[ri]);
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = plane_unit + li * units;
                    if comptime!(lines % units == 0) || line < lines {
                        let i = r * lines + line;
                        view.write(i, view.read(i) * factor);
                    }
                }
            }
        }
    }

    /// Multiply each row by its factor: `self[r, c] *= factors[r]`. The caller syncs on both sides.
    pub fn scale_rows(&mut self, factors: &Tile) {
        let define!(Served) = self.element;
        let cols = comptime!(self.place.space.extent_at(1));
        comptime!(assert!(
            self.place.space.rank() == 2,
            "scale_rows: a rank-2 accumulator tile"
        ));
        let w = self.vector_size();
        let wf = factors.vector_size();
        comptime!(assert!(
            w == 1 && wf == 1,
            "scale_rows: vectorized tiles not supported yet"
        ));
        let total = comptime!(self.place.space.cells());
        let size!(W) = w;
        let size!(WF) = wf;
        let f = factors.flat::<Served, WF>();
        let mut view = self.flat_mut::<Served, W>();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let v = view.read(i).extract(0usize) * f.read(i / cols).extract(0usize);
            view.write(i, Vector::cast_from(v));
            i += workers;
        }
    }

    /// Cast-copy the owned rows into `dest`, laid out in the same lines.
    pub(crate) fn write_rows_to<EA: Float, EP: Numeric>(
        &self,
        dest: &mut Tile,
        state: &RowState<EA>,
    ) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let wp = dest.vector_size();
        comptime!(assert!(
            w == wp,
            "write_rows_to: the probabilities are laid out in the score's lines"
        ));
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let src = self.flat::<EA, W>();
        let mut dst = dest.flat_mut::<EP, W>();

        #[unroll]
        for ri in 0..rpu {
            let r = state.owned_row(ri);
            if r < rows {
                #[unroll]
                for line in 0..lines {
                    let i = r * lines + line;
                    dst.write(i, Vector::<EP, W>::cast_from(src.read(i)));
                }
            }
        }
    }
}
