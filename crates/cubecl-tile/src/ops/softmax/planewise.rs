//! The row ops at plane ownership: a plane owns the rows, its units split the reduced axis.
//! `units` must be the device's committed plane width and `CUBE_DIM_X` whole planes, else
//! the reduction is silently wrong.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// [`scale_and_mask`](Tile::scale_and_mask) at plane ownership.
    pub(crate) fn scale_and_mask_planar<EA: Float>(
        &mut self,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<EA, W>();

        #[unroll]
        for ri in 0..rpp {
            let r = ri;
            if r < rows {
                let q = probe.row_q(r);
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let i = r * lines + line;
                        let mut v = view.read(i) * Vector::<EA, W>::cast_from(scale);
                        #[unroll]
                        for j in 0..w {
                            let masked = probe.masked(q, probe.origin_s + line * w + j, mask);
                            v.insert(j, select(masked, EA::min_value(), v.extract(j)));
                        }
                        view.write(i, v);
                    }
                }
            }
        }
    }

    /// [`row_max`](Tile::row_max) at plane ownership.
    pub(crate) fn row_max_planar<EA: Float>(
        &self,
        acc: &mut Array<EA>,
        base: &Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<EA, W>();

        #[unroll]
        for ri in 0..rpp {
            let mut partial = base[ri];
            let r = ri;
            if r < rows {
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let v = view.read(r * lines + line);
                        #[unroll]
                        for j in 0..w {
                            partial = max(partial, v.extract(j));
                        }
                    }
                }
            }
            acc[ri] =
                comptime!(UnitShare::of_units(units, units)).reduce::<EA>(partial, Monoid::Max);
        }
    }

    /// [`exp_diff`](Tile::exp_diff) at plane ownership.
    pub(crate) fn exp_diff_planar<EA: Float>(
        &mut self,
        rowwise: &Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<EA, W>();

        #[unroll]
        for ri in 0..rpp {
            let r = ri;
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let i = r * lines + line;
                        let mut v = view.read(i);
                        #[unroll]
                        for j in 0..w {
                            v.insert(j, live * (v.extract(j) - safe_m).exp());
                        }
                        view.write(i, v);
                    }
                }
            }
        }
    }

    /// [`row_sum`](Tile::row_sum) at plane ownership. Every unit must contribute its lines once.
    pub(crate) fn row_sum_planar<EA: Float>(
        &self,
        acc: &mut Array<EA>,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<EA, W>();

        #[unroll]
        for ri in 0..rpp {
            let mut partial = EA::from_int(0);
            let r = ri;
            if r < rows {
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let v = view.read(r * lines + line);
                        #[unroll]
                        for j in 0..w {
                            partial += v.extract(j);
                        }
                    }
                }
            }
            acc[ri] =
                comptime!(UnitShare::of_units(units, units)).reduce::<EA>(partial, Monoid::Sum);
        }
    }

    /// [`write_rows_to`](Tile::write_rows_to) at plane ownership.
    pub(crate) fn write_rows_to_planar<EA: Float, EP: Numeric>(
        &self,
        dest: &mut Tile,
        #[comptime] rpp: usize,
        #[comptime] units: usize,
    ) {
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
        for ri in 0..rpp {
            let r = ri;
            if r < rows {
                #[unroll]
                for li in 0..comptime!(lines.div_ceil(units)) {
                    let line = unit(units) + li * units;
                    if comptime!(lines.is_multiple_of(units)) || line < lines {
                        let i = r * lines + line;
                        dst.write(i, Vector::<EP, W>::cast_from(src.read(i)));
                    }
                }
            }
        }
    }
}

/// This unit's index within its plane.
#[cube]
fn unit(#[comptime] units: usize) -> usize {
    UNIT_POS_X as usize % units
}
