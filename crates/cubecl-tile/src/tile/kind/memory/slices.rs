//! A plane's window of shared memory, cut into slices along its innermost axis: the verbs
//! [`AxisSlices`] runs on it, the plane's units taking each slice's cells in turns and meeting on
//! the slice's partials.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Memory {
    /// `self[i, s] = self[i, s] · scale + bias[i, s]` over this window over `space`, `axis` its
    /// innermost.
    pub(crate) fn scale_add_along<E: Float>(
        &mut self,
        scale: E,
        bias: &Procedural,
        #[comptime] space: Space,
        #[comptime] axis: Axis,
    ) {
        let (slices, length) = comptime!((space.slices_along(axis), space.extent(axis)));
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let mut s = UNIT_POS_PLANE as usize;
            while s < length {
                let at = i * length + s;
                let cell = bias.value_at::<E>(i as u32, s as u32, space.clone());
                view.write(
                    at,
                    Vector::cast_from(view.read(at).extract(0usize) * scale + cell),
                );
                s += PLANE_DIM as usize;
            }
        }
    }

    /// Each slice's max over this window over `space`, `axis` its innermost, starting from
    /// `seed`'s.
    pub(crate) fn maxima_along<E: Float>(
        &self,
        seed: &Array<E>,
        #[comptime] space: Space,
        #[comptime] axis: Axis,
    ) -> Array<E> {
        let (slices, length) = comptime!((space.slices_along(axis), space.extent(axis)));
        let size!(W) = 1usize;
        let view = self.flat::<E, W>();
        let mut maxima = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            let mut partial = seed[i];
            let mut s = UNIT_POS_PLANE as usize;
            while s < length {
                partial = max(partial, view.read(i * length + s).extract(0usize));
                s += PLANE_DIM as usize;
            }
            maxima[i] = plane_max(partial);
        }
        maxima
    }

    /// `self[i, s] = 2^(self[i, s] · scale − offsets[i])` over this window over `space`, `axis`
    /// its innermost ([`AxisSlices::exp2_scaled_minus_cell`]).
    pub(crate) fn exp2_scaled_minus_along<E: Float>(
        &mut self,
        scale: E,
        offsets: &Array<E>,
        #[comptime] space: Space,
        #[comptime] axis: Axis,
    ) {
        let (count, length) = comptime!((space.slices_along(axis), space.extent(axis)));
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..count {
            let mut s = UNIT_POS_PLANE as usize;
            while s < length {
                let at = i * length + s;
                let cell = AxisSlices::exp2_scaled_minus_cell::<E>(
                    view.read(at).extract(0usize),
                    scale,
                    offsets[i],
                );
                view.write(at, Vector::cast_from(cell));
                s += PLANE_DIM as usize;
            }
        }
    }

    /// Each slice's sum over this window over `space`, `axis` its innermost.
    pub(crate) fn sums_along<E: Float>(
        &self,
        #[comptime] space: Space,
        #[comptime] axis: Axis,
    ) -> Array<E> {
        let (slices, length) = comptime!((space.slices_along(axis), space.extent(axis)));
        let size!(W) = 1usize;
        let view = self.flat::<E, W>();
        let mut sums = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            let mut partial = E::from_int(0);
            let mut s = UNIT_POS_PLANE as usize;
            while s < length {
                partial += view.read(i * length + s).extract(0usize);
                s += PLANE_DIM as usize;
            }
            sums[i] = plane_sum(partial);
        }
        sums
    }

    /// `self[i, s] *= factors[i]` over this window over `space`, `axis` its innermost.
    pub(crate) fn mul_along<E: Float>(
        &mut self,
        factors: &Array<E>,
        #[comptime] space: Space,
        #[comptime] axis: Axis,
    ) {
        let (length, cells) = comptime!((space.extent(axis), space.cells()));
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<E, W>();
        let mut at = UNIT_POS_PLANE as usize;
        while at < cells {
            view.write(at, view.read(at) * Vector::cast_from(factors[at / length]));
            at += PLANE_DIM as usize;
        }
    }

    /// `self[c] = exp(self[c] · scale + bias[c] − logsumexp[c])` over this window over `space`,
    /// `logsumexp` the cells of a tile it broadcasts over ([`AxisSlices::probabilities`]).
    pub(crate) fn probabilities_along<E: Float>(
        &mut self,
        scale: E,
        bias: &Procedural,
        logsumexp: &Array<E>,
        #[comptime] space: Space,
        #[comptime] broadcast: Broadcast,
    ) {
        let (slices, length) = comptime!((
            space.slices_along(space.axis_at(space.rank() - 1)),
            space.extent_at(space.rank() - 1)
        ));
        let step = comptime!(broadcast.position);
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let first = comptime!(broadcast.at(i, 0));
            let mut s = UNIT_POS_PLANE as usize;
            while s < length {
                let at = i * length + s;
                let score = view.read(at).extract(0usize) * scale
                    + bias.value_at::<E>(i as u32, s as u32, space.clone());
                let probability =
                    AxisSlices::exp_minus_cell::<E>(score, logsumexp[first + s * step]);
                view.write(at, Vector::cast_from(probability));
                s += PLANE_DIM as usize;
            }
        }
    }

    /// `self[c] = probabilities[c] · (self[c] − delta[c])` over this window and `probabilities`,
    /// both over `space`, `delta` the cells of a tile they broadcast over
    /// ([`AxisSlices::softmax_gradient`]).
    pub(crate) fn softmax_gradient_along<E: Float>(
        &mut self,
        probabilities: &Memory,
        delta: &Array<E>,
        #[comptime] space: Space,
        #[comptime] broadcast: Broadcast,
    ) {
        let (slices, length) = comptime!((
            space.slices_along(space.axis_at(space.rank() - 1)),
            space.extent_at(space.rank() - 1)
        ));
        let step = comptime!(broadcast.position);
        let size!(W) = 1usize;
        let read = probabilities.flat::<E, W>();
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let first = comptime!(broadcast.at(i, 0));
            let mut s = UNIT_POS_PLANE as usize;
            while s < length {
                let at = i * length + s;
                let gradient = view.read(at).extract(0usize) - delta[first + s * step];
                view.write(
                    at,
                    Vector::cast_from(read.read(at).extract(0usize) * gradient),
                );
                s += PLANE_DIM as usize;
            }
        }
    }

    /// The `count` cells of this window in order, every unit of its holder reading them all: a
    /// window holding one value a slice of something else, as the rows of a softmax.
    pub(crate) fn cells<E: Float>(&self, #[comptime] count: usize) -> Array<E> {
        let size!(W) = 1usize;
        let view = self.flat::<E, W>();
        let mut cells = Array::<E>::new(count);
        #[unroll]
        for i in 0..count {
            cells[i] = view.read(i).extract(0usize);
        }
        cells
    }

    /// Write `values` over the `count` cells of this window, held by `holder`: every unit writes
    /// its own window, and a plane's first unit the plane's.
    pub(crate) fn set_cells<E: Float>(
        &mut self,
        values: &Array<E>,
        #[comptime] count: usize,
        #[comptime] holder: ComputeScope,
    ) {
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<E, W>();
        if comptime!(holder == ComputeScope::Unit) || UNIT_POS_PLANE == 0 {
            #[unroll]
            for i in 0..count {
                view.write(i, Vector::cast_from(values[i]));
            }
        }
    }
}
