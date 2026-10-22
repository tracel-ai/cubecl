//! The online-softmax step at unit ownership: each op runs over the rows a [`RowState`] owns.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// `self = self * scale`, masked entries driven to `min_value`, per owned row.
    pub(crate) fn scale_and_mask<EA: Float>(
        &mut self,
        scale: EA,
        probe: &MaskProbe,
        mask: &Tile,
        state: &RowState<EA>,
    ) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<EA, W>();

        #[unroll]
        for ri in 0..rpu {
            let r = state.owned_row(ri);
            if r < rows {
                let q = probe.row_q(r);
                #[unroll]
                for line in 0..lines {
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

    /// Per owned row max into `acc`, seeded from the running max `state.m`.
    pub(crate) fn row_max<EA: Float>(&self, acc: &mut Array<EA>, state: &RowState<EA>) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<EA, W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = state.m[ri];
            let r = state.owned_row(ri);
            if r < rows {
                #[unroll]
                for line in 0..lines {
                    let v = view.read(r * lines + line);
                    #[unroll]
                    for j in 0..w {
                        acc[ri] = max(acc[ri], v.extract(j));
                    }
                }
            }
        }
    }

    /// `self = exp(self - rowwise)` per owned row; a fully-masked row goes to zero.
    pub(crate) fn exp_diff<EA: Float>(&mut self, rowwise: &Array<EA>, state: &RowState<EA>) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let threshold = EA::new(LOGIT_MASKED);
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let mut view = self.flat_mut::<EA, W>();

        #[unroll]
        for ri in 0..rpu {
            let r = state.owned_row(ri);
            if r < rows {
                let live = EA::cast_from(rowwise[ri] >= threshold);
                let safe_m = clamp_min(rowwise[ri], threshold);
                #[unroll]
                for line in 0..lines {
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

    /// Per owned row sum into `acc`.
    pub(crate) fn row_sum<EA: Float>(&self, acc: &mut Array<EA>, state: &RowState<EA>) {
        let rpu = comptime!(state.share.rows());
        let rows = comptime!(self.place.space.extent_at(0));
        let cols = comptime!(self.place.space.extent_at(1));
        let w = self.vector_size();
        let size!(W) = w;
        let lines = comptime!(cols / w);
        let view = self.flat::<EA, W>();

        #[unroll]
        for ri in 0..rpu {
            acc[ri] = EA::from_int(0);
            let r = state.owned_row(ri);
            if r < rows {
                #[unroll]
                for line in 0..lines {
                    let v = view.read(r * lines + line);
                    #[unroll]
                    for j in 0..w {
                        acc[ri] += v.extract(j);
                    }
                }
            }
        }
    }
}
