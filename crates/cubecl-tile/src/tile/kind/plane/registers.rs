//! The register-resident accumulator ([`RegisterData`]), the software leaf's counterpart to
//! a cmma fragment.

use cubecl::prelude::*;

use crate::*;

// The block's line width, bound by `alloc` via `register_size`. Allocated at the vector
// element: the CPU backend refuses a scalar array re-viewed as lines.
define_size!(pub(crate) RA);

/// What a register block accumulates: products of two operands under a semiring, or one
/// operand's values under a monoid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Accumulation {
    Contraction(Semiring),
    Reduction(Monoid),
}

impl Accumulation {
    /// The `⊕` partials merge under, and the identity a block opens at.
    pub(crate) fn monoid(self) -> Monoid {
        match self {
            Accumulation::Contraction(semiring) => semiring.add(),
            Accumulation::Reduction(monoid) => monoid,
        }
    }

    /// The semiring a contraction runs under; a reduction runs none.
    pub(crate) fn semiring(self, site: &str) -> Semiring {
        match self {
            Accumulation::Contraction(semiring) => semiring,
            Accumulation::Reduction(monoid) => {
                panic!("{site}: a block opened to reduce under {monoid:?} contracts nothing")
            }
        }
    }
}

/// An `mr × nr` block of `RA`-wide register accumulators, the software [`PlaneTile`].
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct RegisterData {
    /// `mr * nr` lines, each `RA` wide at the element the block holds (width registered in
    /// [`alloc`](Self::alloc)).
    pub(crate) data: Raw<Registers<RA>>,
    /// Physical line width, the numeric twin of `RA`; comptime, for the line arithmetic.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Contracted values one step of the rhs brings for one cell: `1` for an rhs lined along the
    /// block's columns, else the run a load holds of a column, which each step sums into its
    /// cell, so the block is one value wide, or, [opened for partials](RegisterBlock::partials),
    /// keeps a lane each of, so a line is one cell `contracted_per_step` wide.
    #[cube(comptime)]
    pub(crate) contracted_per_step: usize,
    /// Rows in the block.
    #[cube(comptime)]
    pub(crate) mr: usize,
    /// Lines per row: `n / vector_size`, or `n` where a line holds one cell's partials.
    #[cube(comptime)]
    pub(crate) nr: usize,
    /// The sink's matrix this block was sized against.
    #[cube(comptime)]
    pub(crate) axes: MatrixAxes,
    /// Execution configuration for this register leaf.
    #[cube(comptime)]
    pub(crate) config: RegisterBlock,
    /// What this block accumulates, and so the `⊕` its partials merge under.
    #[cube(comptime)]
    pub(crate) accumulation: Accumulation,
}

/// Whether a block of `vector_size`-wide lines contracting `contracted_per_step` values a step
/// holds each run's products a lane each, one cell a line, rather than whole cells.
pub(crate) fn holds_partials(contracted_per_step: usize, vector_size: usize) -> bool {
    contracted_per_step > 1 && vector_size > 1
}

/// The cells one `vector_size`-wide line holds: one where it holds a cell's partials.
fn cells_per_line(contracted_per_step: usize, vector_size: usize) -> usize {
    match holds_partials(contracted_per_step, vector_size) {
        true => 1,
        false => vector_size,
    }
}

/// Panics where a block of `vector_size`-wide lines contracting `contracted_per_step` values a
/// step holds a cell's partials: its cells are whole only at the drain, and `site` reads them
/// before.
pub(crate) fn assert_whole_cells(contracted_per_step: usize, vector_size: usize, site: &str) {
    assert!(
        !holds_partials(contracted_per_step, vector_size),
        "{site}: the block holds each cell as {vector_size} partials, summed only at its drain; \
         open it without `RegisterBlock::partials` to read its cells"
    );
}

/// Bind the block width `RA` for what follows. A kernel may hold blocks of several widths, as an
/// attention unit holds a score one value a cell beside its output's lines, so every block binds
/// its own before it touches its lines.
#[cube]
pub(crate) fn register_block_size(#[comptime] vector_size: usize) {
    intrinsic!(|scope| {
        scope.register_size::<RA>(vector_size);
    });
}

#[cube]
impl RegisterData {
    /// An uninitialized `m × n` block of `vector_size`-wide lines, contracting an rhs that brings
    /// `contracted_per_step` values of a cell a step. `n` must be a whole number of lines; a line
    /// `contracted_per_step` wide holds one cell's partials, so then `n` is any width.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alloc<T: Numeric>(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] vector_size: usize,
        #[comptime] contracted_per_step: usize,
        #[comptime] config: RegisterBlock,
        #[comptime] accumulation: Accumulation,
    ) -> RegisterData {
        comptime!(assert!(
            contracted_per_step == 1 || vector_size == 1 || vector_size == contracted_per_step,
            "RegisterData::alloc: a block contracting {contracted_per_step} values of a cell a step sums them into \
             the cell, one value wide, or keeps them a lane each, {contracted_per_step} wide; not {vector_size}"
        ));
        let cells = comptime!(cells_per_line(contracted_per_step, vector_size));
        comptime!(assert!(
            vector_size > 0 && n.is_multiple_of(cells),
            "RegisterData::alloc: n ({n}) must be a whole number of {vector_size}-wide lines"
        ));
        register_block_size(vector_size);
        let nr = comptime!(n / cells);
        RegisterData {
            data: Raw::<Registers<RA>>::of::<T>(Array::<Vector<T, RA>>::new(comptime!(m * nr))),
            vector_size,
            contracted_per_step,
            mr: m,
            nr,
            axes,
            config,
            accumulation,
        }
    }

    pub(crate) fn zero<T: Numeric>(&mut self) {
        self.init::<T>(T::from_int(0));
    }

    /// Every cell at `val`. A cell held as partials takes it in its first lane and the identity
    /// in the rest, so its lanes sum to `val`.
    pub(crate) fn init<T: Numeric>(&mut self, val: T) {
        register_block_size(self.vector_size);
        let data = self.data.at_mut::<T>();
        let count = comptime!(self.mr * self.nr);
        let mut line = Vector::<T, RA>::cast_from(val);
        if comptime!(holds_partials(self.contracted_per_step, self.vector_size)) {
            line = Vector::<T, RA>::cast_from(Monoid::identity::<T>(comptime!(
                self.accumulation.monoid()
            )));
            line.insert(0usize, val);
        }
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            data[i] = line;
        }
    }

    /// Add `src`'s block (same grid and values a step) into this one, casting each line up to `T`.
    pub(crate) fn add_cast_from<T: Numeric, S: Numeric>(&mut self, src: &RegisterData) {
        let data = self.data.at_mut::<T>();
        comptime!(assert!(
            self.mr == src.mr
                && self.nr == src.nr
                && self.contracted_per_step == src.contracted_per_step
                && self.vector_size == src.vector_size,
            "RegisterData::add_cast_from: the blocks are {}x{} over {} contracted and {}x{} over \
             {}; a promotion adds line for line and both are opened against one sink",
            self.mr,
            self.nr,
            self.contracted_per_step,
            src.mr,
            src.nr,
            src.contracted_per_step
        ));
        register_block_size(self.vector_size);
        let count = comptime!(self.mr * self.nr);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            data[i] += Vector::<T, RA>::cast_from(src.data.at::<S>()[i]);
        }
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the block's columns, in the
    /// registers that hold it.
    pub(crate) fn mul_along<T: Numeric>(&mut self, factors: &Array<T>, #[comptime] first: usize) {
        register_block_size(self.vector_size);
        let data = self.data.at_mut::<T>();
        #[unroll]
        for r in 0..self.mr {
            let factor = Vector::<T, RA>::cast_from(factors[first + r]);
            #[unroll]
            for n in 0..self.nr {
                data[r * self.nr + n] *= factor;
            }
        }
    }

    /// Multiply every partial this block holds by `factor`.
    pub(crate) fn scale<T: Numeric>(&mut self, factor: T) {
        register_block_size(self.vector_size);
        let data = self.data.at_mut::<T>();
        let count = comptime!(self.mr * self.nr);
        let line = Vector::<T, RA>::cast_from(factor);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            data[i] *= line;
        }
    }
}

#[cube]
impl RegisterData {
    /// Write the block into `mem`'s window, casting down to its element. A block of partials sums
    /// each line's lanes into its cell on the way out, into a sink one value a cell.
    pub(crate) fn store_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: Space,
    ) {
        register_block_size(self.vector_size);
        let size!(A) = comptime!(cells_per_line(self.contracted_per_step, self.vector_size));
        self.drain::<T, Out, A>(mem, space);
    }

    /// The cell a line drains to: the line, or the sum of its partials.
    fn drained<T: Numeric, Out: Numeric, A: Size>(&self, line: Vector<T, RA>) -> Vector<Out, A> {
        if comptime!(holds_partials(self.contracted_per_step, self.vector_size)) {
            let monoid = comptime!(self.accumulation.monoid());
            Vector::<Out, A>::cast_from(Monoid::reduce::<T, RA>(line, self.vector_size, monoid))
        } else {
            Vector::<Out, A>::cast_from(line)
        }
    }

    /// [`store_cast_window`](Self::store_cast_window) into a sink of `A`-wide cells.
    fn drain<T: Numeric, Out: Numeric, A: Size>(&self, mem: &mut Memory, #[comptime] space: Space) {
        let data = self.data.at::<T>();
        let mem_write = comptime!(mem.access.write);
        let unit_share = comptime!(mem.unit_share);
        let monoid = comptime!(self.accumulation.monoid());
        let mut sink = mem.matrix_mut::<Out, A>(0usize, comptime!(self.axes), space);

        // Split at comptime: a per-line `match` plus a unit guard breaks the CPU backend.
        match comptime!(Drain::of(unit_share, mem_write)) {
            Drain::EachUnit =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let cell = self.drained::<T, Out, A>(data[comptime!(i * self.nr + n)]);
                        sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                    }
                }
            }
            Drain::UnitZero =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let cell = self.drained::<T, Out, A>(data[comptime!(i * self.nr + n)]);
                        if UNIT_POS_X == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
            Drain::PlaneReduce =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = UnitShare::Plane
                            .reduce::<Vector<T, RA>>(data[comptime!(i * self.nr + n)], monoid);
                        let cell = self.drained::<T, Out, A>(combined);
                        if UNIT_POS_X == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
            Drain::GroupReduce { unit_bits } =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = comptime!(UnitShare::Group { unit_bits })
                            .reduce::<Vector<T, RA>>(data[comptime!(i * self.nr + n)], monoid);
                        let cell = self.drained::<T, Out, A>(combined);
                        let unit_in_group = UNIT_POS_X & comptime!(unit_bits as u32);
                        if unit_in_group == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
        }
    }
}

/// The block cut into slices along its columns, each a unit holds whole: the verbs
/// [`AxisSlices`] runs on a register block. The block holds whole cells, never a line of partials
/// of one ([`assert_whole_cells`]).
#[cube]
impl RegisterData {
    /// `self[i, s] = self[i, s] · scale + bias[i, s]` at every cell, the bias read at the cell of
    /// a tile over `space` that the block holds.
    pub(crate) fn scale_add_along<E: Float>(
        &mut self,
        scale: E,
        bias: &Procedural,
        #[comptime] space: Space,
    ) {
        let data = self.data.at_mut::<E>();
        #[unroll]
        for r in 0..self.mr {
            #[unroll]
            for n in 0..self.nr {
                let at = r * self.nr + n;
                let mut line = data[at];
                #[unroll]
                for j in 0..self.vector_size {
                    let column = n * self.vector_size + j;
                    let cell = bias.value_at::<E>(r as u32, column as u32, space.clone());
                    line.insert(j, line.extract(j) * scale + cell);
                }
                data[at] = line;
            }
        }
    }

    /// Each slice's max, starting from `seed`'s.
    pub(crate) fn maxima_along<E: Float>(&self, seed: &Array<E>) -> Array<E> {
        let data = self.data.at::<E>();
        let mut maxima = Array::<E>::new(self.mr);
        #[unroll]
        for r in 0..self.mr {
            let mut slice = seed[r];
            #[unroll]
            for n in 0..self.nr {
                let line = data[r * self.nr + n];
                #[unroll]
                for j in 0..self.vector_size {
                    slice = max(slice, line.extract(j));
                }
            }
            maxima[r] = slice;
        }
        maxima
    }

    /// `self[i, s] = 2^(self[i, s] · scale − offsets[i])` ([`AxisSlices::exp2_scaled_minus_cell`]).
    pub(crate) fn exp2_scaled_minus_along<E: Float>(&mut self, scale: E, offsets: &Array<E>) {
        let data = self.data.at_mut::<E>();
        #[unroll]
        for r in 0..self.mr {
            #[unroll]
            for n in 0..self.nr {
                let at = r * self.nr + n;
                let mut line = data[at];
                #[unroll]
                for j in 0..self.vector_size {
                    line.insert(
                        j,
                        AxisSlices::exp2_scaled_minus_cell::<E>(line.extract(j), scale, offsets[r]),
                    );
                }
                data[at] = line;
            }
        }
    }

    /// `self[c] = exp(self[c] · scale + bias[c] − logsumexp[c])` at every cell, the bias read at
    /// the cell of a tile over `space` that the block holds, `logsumexp` the cells of a tile it
    /// broadcasts over ([`AxisSlices::probabilities`]).
    pub(crate) fn probabilities_along<E: Float>(
        &mut self,
        scale: E,
        bias: &Procedural,
        logsumexp: &Array<E>,
        #[comptime] space: Space,
        #[comptime] broadcast: Broadcast,
    ) {
        let data = self.data.at_mut::<E>();
        #[unroll]
        for r in 0..self.mr {
            #[unroll]
            for n in 0..self.nr {
                let at = r * self.nr + n;
                let mut line = data[at];
                #[unroll]
                for j in 0..self.vector_size {
                    let column = n * self.vector_size + j;
                    let score = line.extract(j) * scale
                        + bias.value_at::<E>(r as u32, column as u32, space.clone());
                    let slot = comptime!(broadcast.at(r, n * self.vector_size + j));
                    line.insert(j, AxisSlices::exp_minus_cell::<E>(score, logsumexp[slot]));
                }
                data[at] = line;
            }
        }
    }

    /// `self[c] = probabilities[c] · (self[c] − delta[c])` ([`AxisSlices::softmax_gradient`]),
    /// `probabilities` a block of this one's shape, `delta` the cells of a tile both broadcast
    /// over.
    pub(crate) fn softmax_gradient_along<E: Float>(
        &mut self,
        probabilities: &RegisterData,
        delta: &Array<E>,
        #[comptime] broadcast: Broadcast,
    ) {
        comptime!(assert!(
            self.mr == probabilities.mr
                && self.nr == probabilities.nr
                && self.vector_size == probabilities.vector_size,
            "RegisterData::softmax_gradient_along: the probabilities' block is {}×{} lines of {}, \
             the gradient's {}×{} lines of {}",
            probabilities.mr,
            probabilities.nr,
            probabilities.vector_size,
            self.mr,
            self.nr,
            self.vector_size
        ));
        let read = probabilities.data.at::<E>();
        let data = self.data.at_mut::<E>();
        #[unroll]
        for r in 0..self.mr {
            #[unroll]
            for n in 0..self.nr {
                let at = r * self.nr + n;
                let mut line = data[at];
                let probability = read[at];
                #[unroll]
                for j in 0..self.vector_size {
                    let slot = comptime!(broadcast.at(r, n * self.vector_size + j));
                    line.insert(j, probability.extract(j) * (line.extract(j) - delta[slot]));
                }
                data[at] = line;
            }
        }
    }

    /// Each slice's sum.
    pub(crate) fn sums_along<E: Float>(&self) -> Array<E> {
        let data = self.data.at::<E>();
        let mut sums = Array::<E>::new(self.mr);
        #[unroll]
        for r in 0..self.mr {
            let mut slice = E::from_int(0);
            #[unroll]
            for n in 0..self.nr {
                let line = data[r * self.nr + n];
                #[unroll]
                for j in 0..self.vector_size {
                    slice += line.extract(j);
                }
            }
            sums[r] = slice;
        }
        sums
    }
}
