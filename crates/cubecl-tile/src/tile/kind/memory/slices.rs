//! A plane's window of shared memory, cut into slices along its innermost dimension: the verbs
//! [`DimensionSlices`] runs on it, the plane's units taking each slice's cells in turns and meeting
//! on the slice's partials.

use cubecl::prelude::*;

use crate::*;

#[cube]
impl Memory {
    /// `self[i, s] = self[i, s] · scale + bias[i, s]` over this window over `space`, `dimension`
    /// its innermost.
    pub(crate) fn scale_add_along<E: Float>(
        &mut self,
        scale: E,
        bias: &Procedural,
        #[comptime] space: &Space,
        #[comptime] dimension: Dimension,
    ) {
        let (slices, length) = comptime!((space.slices_along(dimension), space.extent(dimension)));
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let at = i * loads + l;
                let mut line = view.read(at);
                #[unroll]
                for lane in 0..width {
                    let cell = bias.value_at::<E>(i as u32, (l * width + lane) as u32, space);
                    line.insert(lane, line.extract(lane) * scale + cell);
                }
                view.write(at, line);
                l += PLANE_DIM as usize;
            }
        }
    }

    /// `self[i, s] = self[i, s] · exp(bias[i, s])` over this window over `space`, `dimension`
    /// its innermost ([`DimensionSlices::mul_exp`]).
    pub(crate) fn mul_exp_along<E: Float>(
        &mut self,
        bias: &Procedural,
        #[comptime] space: &Space,
        #[comptime] dimension: Dimension,
    ) {
        let (slices, length) = comptime!((space.slices_along(dimension), space.extent(dimension)));
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let at = i * loads + l;
                let mut line = view.read(at);
                #[unroll]
                for lane in 0..width {
                    let cell = bias.value_at::<E>(i as u32, (l * width + lane) as u32, space);
                    line.insert(
                        lane,
                        DimensionSlices::mul_exp_cell::<E>(line.extract(lane), cell),
                    );
                }
                view.write(at, line);
                l += PLANE_DIM as usize;
            }
        }
    }

    /// Each slice's max over this window over `space`, `dimension` its innermost, starting from
    /// `seed`'s.
    pub(crate) fn maxima_along<E: Float>(
        &self,
        seed: &Array<E>,
        #[comptime] space: &Space,
        #[comptime] dimension: Dimension,
    ) -> Array<E> {
        let (slices, length) = comptime!((space.slices_along(dimension), space.extent(dimension)));
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let view = self.flat::<E, W>();
        let mut maxima = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            let mut partial = seed[i];
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let line = view.read(i * loads + l);
                #[unroll]
                for lane in 0..width {
                    partial = max(partial, line.extract(lane));
                }
                l += PLANE_DIM as usize;
            }
            maxima[i] = plane_max(partial);
        }
        maxima
    }

    /// `self[i, s] = 2^(self[i, s] · scale − offsets[i])` over this window over `space`,
    /// `dimension` its innermost ([`DimensionSlices::exp2_scaled_minus_cell`]).
    pub(crate) fn exp2_scaled_minus_along<E: Float>(
        &mut self,
        scale: E,
        offsets: &Array<E>,
        #[comptime] space: &Space,
        #[comptime] dimension: Dimension,
    ) {
        let (count, length) = comptime!((space.slices_along(dimension), space.extent(dimension)));
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..count {
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let at = i * loads + l;
                let mut line = view.read(at);
                #[unroll]
                for lane in 0..width {
                    line.insert(
                        lane,
                        DimensionSlices::exp2_scaled_minus_cell::<E>(
                            line.extract(lane),
                            scale,
                            offsets[i],
                        ),
                    );
                }
                view.write(at, line);
                l += PLANE_DIM as usize;
            }
        }
    }

    /// Each slice's sum over this window over `space`, `dimension` its innermost.
    pub(crate) fn sums_along<E: Float>(
        &self,
        #[comptime] space: &Space,
        #[comptime] dimension: Dimension,
    ) -> Array<E> {
        let (slices, length) = comptime!((space.slices_along(dimension), space.extent(dimension)));
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let view = self.flat::<E, W>();
        let mut sums = Array::<E>::new(slices);
        #[unroll]
        for i in 0..slices {
            let mut partial = E::from_int(0);
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let line = view.read(i * loads + l);
                #[unroll]
                for lane in 0..width {
                    partial += line.extract(lane);
                }
                l += PLANE_DIM as usize;
            }
            sums[i] = plane_sum(partial);
        }
        sums
    }

    /// `self[i, s] *= factors[i]` over this window over `space`, `dimension` its innermost.
    pub(crate) fn mul_along<E: Float>(
        &mut self,
        factors: &Array<E>,
        #[comptime] space: &Space,
        #[comptime] dimension: Dimension,
    ) {
        let (length, cells) = comptime!((space.extent(dimension), space.cells()));
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let mut view = self.flat_mut::<E, W>();
        let mut at = UNIT_POS_PLANE as usize;
        while at < cells / width {
            view.write(at, view.read(at) * Vector::cast_from(factors[at / loads]));
            at += PLANE_DIM as usize;
        }
    }

    /// `self[c] = exp(self[c] · scale + bias[c] − logsumexp[c])` over this window over `space`,
    /// `logsumexp` the cells of a tile it broadcasts over ([`DimensionSlices::probabilities`]).
    pub(crate) fn probabilities_along<E: Float>(
        &mut self,
        scale: E,
        bias: &Procedural,
        logsumexp: &Array<E>,
        #[comptime] space: &Space,
        #[comptime] broadcast: Broadcast,
    ) {
        let (slices, length) = comptime!((
            space.slices_along(space.dimension_at(space.rank() - 1)),
            space.extent_at(space.rank() - 1)
        ));
        let step = broadcast.position;
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let first = broadcast.at(i, 0);
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let at = i * loads + l;
                let mut line = view.read(at);
                #[unroll]
                for lane in 0..width {
                    let s = l * width + lane;
                    let score =
                        line.extract(lane) * scale + bias.value_at::<E>(i as u32, s as u32, space);
                    line.insert(
                        lane,
                        DimensionSlices::exp_minus_cell::<E>(score, logsumexp[first + s * step]),
                    );
                }
                view.write(at, line);
                l += PLANE_DIM as usize;
            }
        }
    }

    /// `self[c] = probabilities[c] · (self[c] − delta[c])` over this window and `probabilities`,
    /// both over `space`, `delta` the cells of a tile they broadcast over
    /// ([`DimensionSlices::softmax_gradient`]).
    pub(crate) fn softmax_gradient_along<E: Float>(
        &mut self,
        probabilities: &Memory,
        delta: &Array<E>,
        #[comptime] space: &Space,
        #[comptime] broadcast: Broadcast,
    ) {
        let (slices, length) = comptime!((
            space.slices_along(space.dimension_at(space.rank() - 1)),
            space.extent_at(space.rank() - 1)
        ));
        let step = broadcast.position;
        let width = comptime!(self.store.vector_size);
        let loads = self.loads_along(length);
        let size!(W) = width;
        let read = probabilities.flat::<E, W>();
        let mut view = self.flat_mut::<E, W>();
        #[unroll]
        for i in 0..slices {
            let first = broadcast.at(i, 0);
            let mut l = UNIT_POS_PLANE as usize;
            while l < loads {
                let at = i * loads + l;
                let mut line = view.read(at);
                let probability = read.read(at);
                #[unroll]
                for lane in 0..width {
                    let gradient = line.extract(lane) - delta[first + (l * width + lane) * step];
                    line.insert(lane, probability.extract(lane) * gradient);
                }
                view.write(at, line);
                l += PLANE_DIM as usize;
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
