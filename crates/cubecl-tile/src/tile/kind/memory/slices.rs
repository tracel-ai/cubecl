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

    /// `self[i, s] = exp(self[i, s] − slices[i])` over this window over `space`, `axis` its
    /// innermost ([`AxisSlices::exp_minus_cell`]).
    pub(crate) fn exp_minus_along<E: Float>(
        &mut self,
        slices: &Array<E>,
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
                let cell =
                    AxisSlices::exp_minus_cell::<E>(view.read(at).extract(0usize), slices[i]);
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

    /// The `count` cells of this window in order, every unit of the plane reading them all: a
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

    /// Write `values` over the `count` cells of this window, from the plane's first unit.
    pub(crate) fn set_cells<E: Float>(&mut self, values: &Array<E>, #[comptime] count: usize) {
        let size!(W) = 1usize;
        let mut view = self.flat_mut::<E, W>();
        if UNIT_POS_PLANE == 0 {
            #[unroll]
            for i in 0..count {
                view.write(i, Vector::cast_from(values[i]));
            }
        }
    }
}
