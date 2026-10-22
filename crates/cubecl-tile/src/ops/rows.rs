//! Row verbs on a tile. No op here syncs; the caller owns the barriers.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
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
}
