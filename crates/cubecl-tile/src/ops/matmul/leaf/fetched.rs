//! A register block's contraction split around its reads: what one contraction loads, fetched
//! into the unit's registers undecoded ([`Fetched`]), and the contraction that decodes and
//! multiplies it later ([`RegisterData::mma_fetched`]). A walk fetching the next step before it
//! contracts this one keeps that step's loads in flight behind this step's decode and multiplies
//! ([`Tile::mma_walk`]).

use cubecl::prelude::*;

use crate::*;

// The widths a fetch holds its loads at, bound by `Fetched::new`: the lhs's line, the rhs's
// words a load, and the words a load of its block scales reads.
define_size!(pub(crate) FL);
define_size!(pub(crate) FW);
define_size!(pub(crate) FS);

/// Bind the widths a [`Fetched`] holds its loads at.
#[cube]
fn register_fetched_widths(
    #[comptime] lhs: usize,
    #[comptime] words: usize,
    #[comptime] scales: usize,
) {
    intrinsic!(|scope| {
        scope.register_size::<FL>(lhs);
        scope.register_size::<FW>(words);
        scope.register_size::<FS>(scales);
    });
}

/// What one register-block contraction along its rhs's runs reads, held in a unit's registers as
/// it was loaded: the lhs's lines, the rhs's loads as the words they read, and for each of the
/// rhs's lines the load of its block scale and where in it the scale lies. Nothing in it is
/// decoded, so filling it waits on no load.
#[derive(CubeType)]
pub(crate) struct Fetched {
    /// `lines × mr` lhs lines, at the element the lhs is stored at.
    lhs: Raw<Registers<FL>>,
    /// `lines × groups` rhs loads, each the words of one group of columns' runs.
    words: Array<Vector<u32, FW>>,
    /// `lines × nr` loads of the rhs's block scales, one a column's run.
    scales: Array<Vector<u32, FS>>,
    /// Where each of those scales lies in its load.
    scale_at: Array<u32>,
}

/// The comptime shape of one register-block contraction along the rhs's runs.
#[derive(Clone, Debug)]
pub(crate) struct FetchedShape {
    /// Rows, and columns, of the block.
    mr: usize,
    nr: usize,
    /// Columns one rhs load holds a run of each.
    columns: usize,
    /// Values one run holds: what each step contracts for one cell.
    run: usize,
    /// The lhs's line width.
    lw: usize,
    /// Values one rhs load holds.
    load: usize,
    /// Words one rhs load reads.
    words: usize,
    /// Lines of the contraction: runs along `k` the block contracts in one call.
    lines: usize,
    lhs_axes: MatrixAxes,
    rhs_axes: MatrixAxes,
    acc_axes: MatrixAxes,
    /// The innermost scales' words a load, and the field one scale is; `None` without scales.
    scales: Option<(usize, Field)>,
    /// The field one rhs value is.
    field: Field,
}

impl FetchedShape {
    /// rhs loads a line of the contraction.
    fn groups(&self) -> usize {
        self.nr / self.columns
    }
}

#[cube]
impl RegisterData {
    /// The shape of this block's contraction of `lhs` and `rhs` into `out` from fetched loads, which
    /// refuses any contraction a fetch does not hold: a block whose cells sum a run of the rhs's
    /// along `k`, against a packed rhs in memory under packed block scales,
    /// and a plain, unscaled lhs in memory. Scales outside the innermost are the reader's coarse
    /// factor, met once.
    fn fetched_shape(
        &self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: Space,
    ) -> comptime_type!(FetchedShape) {
        let rhs_load = rhs.vector_tile();
        let run = comptime!(rhs_load.run_length());
        let lw = lhs.vector_size();
        let lhs_packing = lhs.packing();
        let rhs_packing = rhs.packing();
        let lhs_scaled = lhs.factor.scaled();
        let rhs_factor = rhs.factor.clone();
        let depth = rhs_factor.depth();
        let words = rhs.words_a_line();
        let scales = if comptime!(depth == 0) {
            comptime!(None)
        } else {
            let scale = rhs_factor.innermost_scale();
            let scale_packing = scale.packing();
            let scale_words = scale.words_a_line();
            comptime!(match scale_packing {
                Packing::Packed { field } => Some((scale_words, field)),
                Packing::Plain | Packing::Words { .. } => panic!(
                    "RegisterData::fetch: a fetch holds packed block scales as words; these \
                     are stored plain"
                ),
            })
        };
        let lhs_in_memory = lhs.is_memory();
        let rhs_in_memory = rhs.is_memory();
        let in_memory = comptime!(lhs_in_memory && rhs_in_memory);
        comptime!({
            assert!(
                in_memory,
                "RegisterData::fetch: a fetch reads its operands from memory"
            );
            assert!(
                lhs_packing == Packing::Plain && !lhs_scaled,
                "RegisterData::fetch: a fetch holds a plain, unscaled lhs"
            );
            let field = match rhs_packing {
                Packing::Packed { field } => field,
                Packing::Plain | Packing::Words { .. } => {
                    panic!("RegisterData::fetch: a fetch holds a packed rhs as its words")
                }
            };
            let along_k = lhs
                .place
                .space
                .contains(rhs.place.space.axis_at(rhs.place.space.rank() - 1));
            assert!(
                along_k && self.contracted_per_step == run && lw == run,
                "RegisterData::fetch: a fetch holds a contraction along the rhs's runs, its \
                 cells summing a run of {run}, its lhs lined along it"
            );
            let operands = Space::merge(&[&lhs.place.space, &rhs.place.space]);
            let kc = operands.contracted_extent(&out);
            assert!(
                kc.is_multiple_of(run),
                "RegisterData::fetch: a step of {kc} is not whole runs of {run}"
            );
            let acc_axes = MatrixAxes::accumulator(&out, &lhs.place.space);
            let cols = acc_axes.cols(&out);
            let (mr, nr) = (self.mr, self.nr);
            FetchedShape {
                mr,
                nr,
                columns: rhs_load.values() / run,
                run,
                lw,
                load: rhs_load.values(),
                words,
                lines: kc / run,
                lhs_axes: MatrixAxes::new(&lhs.place.space, mr, kc)
                    .unwrap_or_else(|e| panic!("{e}")),
                rhs_axes: MatrixAxes::new(&rhs.place.space, cols, kc)
                    .unwrap_or_else(|e| panic!("{e}")),
                acc_axes,
                scales,
                field,
            }
        })
    }

    /// Registers for one fetch of `lhs` and `rhs` contracted into `out` by this block.
    pub(crate) fn fetched<EL: Numeric>(
        &self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: Space,
    ) -> Fetched {
        let shape = self.fetched_shape(lhs, rhs, out);
        let scale_words = comptime!(shape.scales.map_or(1, |(words, _)| words));
        register_fetched_widths(shape.lw, shape.words, scale_words);
        let scales = comptime!(shape.lines * shape.nr);
        Fetched {
            lhs: Raw::<Registers<FL>>::of::<EL>(Array::<Vector<EL, FL>>::new(comptime!(
                shape.lines * shape.mr
            ))),
            words: Array::<Vector<u32, FW>>::new(comptime!(shape.lines * shape.groups())),
            scales: Array::<Vector<u32, FS>>::new(scales),
            scale_at: Array::<u32>::new(scales),
        }
    }

    /// Load what this block's contraction of `lhs` and `rhs` into `out` reads into `fetched`,
    /// undecoded.
    pub(crate) fn fetch<EL: Numeric>(
        &self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: Space,
        fetched: &mut Fetched,
    ) {
        let shape = self.fetched_shape(lhs, rhs, comptime!(out.clone()));
        let scale_words = comptime!(shape.scales.map_or(1, |(words, _)| words));
        register_fetched_widths(shape.lw, shape.words, scale_words);
        let (mr, nr, lines, groups) = comptime!((shape.mr, shape.nr, shape.lines, shape.groups()));
        let lhs_mat = lhs.matrix_packed::<EL, FL>(comptime!(shape.lhs_axes), 0usize);
        let rhs_words = rhs.matrix_words::<FW>(comptime!(shape.rhs_axes), 0usize);
        let lhs_lines = fetched.lhs.at_mut::<EL>();
        #[unroll]
        for line in 0..lines {
            #[unroll]
            for i in 0..mr {
                lhs_lines[line * mr + i] =
                    lhs_mat.read(((i as u32).runtime(), (line as u32).runtime()));
            }
            #[unroll]
            for g in 0..groups {
                fetched.words[line * groups + g] =
                    rhs_words.read(((g as u32).runtime(), (line as u32).runtime()));
            }
        }
        if comptime!(shape.scales.is_some()) {
            let reader = rhs.reader(
                comptime!(shape.rhs_axes),
                0usize,
                comptime!(Side::Rhs),
                comptime!(out.clone()),
                comptime!(shape.acc_axes),
            );
            let scale = reader.inner.innermost_scale();
            #[unroll]
            for line in 0..lines {
                #[unroll]
                for n in 0..nr {
                    let coords = TileMatrix::value_coords(
                        (n as u32).runtime(),
                        (line as u32).runtime(),
                        reader.matrix,
                        comptime!(&reader.values),
                        comptime!(reader.axes),
                        comptime!(reader.vector_size),
                    );
                    let own = scale.scale_coords(&coords, &reader.values);
                    fetched.scales[line * nr + n] =
                        scale.words_holding::<FS>(&own, comptime!(Guard::Checked));
                    fetched.scale_at[line * nr + n] = scale.position_in_line(&own);
                }
            }
        }
    }

    /// `self += lhs · rhs` from what [`fetch`](RegisterData::fetch) loaded of them into `fetched`:
    /// each rhs load decoded, each column's run summed against the lhs's line into its cell, and
    /// the sum multiplied by the run's scale.
    pub(crate) fn mma_fetched<T: Numeric, EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: Space,
        fetched: &Fetched,
    ) {
        let shape = self.fetched_shape(lhs, rhs, comptime!(out.clone()));
        let semiring = comptime!(self.accumulation.semiring("RegisterData::mma_fetched"));
        register_block_size(self.vector_size);
        let scale_words = comptime!(shape.scales.map_or(1, |(words, _)| words));
        register_fetched_widths(shape.lw, shape.words, scale_words);
        let (mr, nr, lines, groups, columns, run) = comptime!((
            shape.mr,
            shape.nr,
            shape.lines,
            shape.groups(),
            shape.columns,
            shape.run
        ));
        let size!(RL) = comptime!(shape.load);
        let size!(L) = comptime!(shape.lw);
        // A cell one value wide takes each run's sum, and under the sum of products the run's
        // scale multiplies that sum, once a cell; a cell as wide as the run keeps its products a
        // lane each, scaled as they are read.
        let summed = comptime!(self.vector_size == 1);
        let scale_sum = comptime!(summed && semiring == Semiring::SUM_PROD);
        let reader = rhs.reader(
            comptime!(shape.rhs_axes),
            0usize,
            comptime!(Side::Rhs),
            comptime!(out.clone()),
            comptime!(shape.acc_axes),
        );
        let data = self.data.at_mut::<T>();
        let lhs_lines = fetched.lhs.at::<EL>();
        #[unroll]
        for line in 0..lines {
            let mut a = Array::<Vector<T, L>>::new(mr);
            #[unroll]
            for i in 0..mr {
                a[i] = Vector::<T, L>::cast_from(lhs_lines[line * mr + i]);
            }
            #[unroll]
            for g in 0..groups {
                let held = unpack_line::<ER, FW, RL>(
                    fetched.words[line * groups + g],
                    comptime!(shape.field),
                );
                #[unroll]
                for j in 0..columns {
                    let n = comptime!(g * columns + j);
                    let scale =
                        fetched_scale(&reader, &rhs.factor, fetched, comptime!(line * nr + n));
                    let values = values_at::<T, ER, RL, L>(
                        held,
                        comptime!(j * run),
                        comptime!(Packing::Plain),
                    );
                    let values = if comptime!(scale_sum) {
                        values
                    } else {
                        scaled_by::<T, L>(values, scale)
                    };
                    #[unroll]
                    for i in 0..mr {
                        let at = comptime!(i * nr + n);
                        if comptime!(summed) {
                            let mut sum = Vector::<T, RA>::cast_from(Monoid::identity::<T>(
                                comptime!(semiring.add()),
                            ));
                            #[unroll]
                            for v in 0..run {
                                sum = semiring.step::<Vector<T, RA>>(
                                    Vector::<T, RA>::cast_from(a[i].extract(v)),
                                    Vector::<T, RA>::cast_from(values.extract(v)),
                                    sum,
                                );
                            }
                            if comptime!(scale_sum) {
                                sum = scaled_by::<T, RA>(sum, scale);
                            }
                            data[at] =
                                comptime!(semiring.add()).combine::<Vector<T, RA>>(data[at], sum);
                        } else {
                            data[at] = semiring.step::<Vector<T, RA>>(
                                Vector::<T, RA>::cast_from(a[i]),
                                Vector::<T, RA>::cast_from(values),
                                data[at],
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The scale [`fetch`](RegisterData::fetch) held the load of at `index`, decoded, times every
/// coarser level the reader met: `1` times them where the values carry no block scale.
#[cube]
fn fetched_scale(
    reader: &FactorReader,
    factor: &Factor,
    fetched: &Fetched,
    #[comptime] index: usize,
) -> f32 {
    if comptime!(reader.scaled) {
        let depth = factor.depth();
        if comptime!(depth == 0) {
            reader.coarse
        } else {
            let scale = factor.innermost_scale();
            let one =
                scale.value_in_words::<f32, FS>(fetched.scales[index], fetched.scale_at[index]);
            one * reader.coarse
        }
    } else {
        1.0f32.runtime()
    }
}

/// `values` times `scale`, multiplied in `f32` and rounded to `E` once, as
/// [`FactorReader::apply_at`] does.
#[cube]
fn scaled_by<E: Numeric, V: Size>(values: Vector<E, V>, scale: f32) -> Vector<E, V> {
    let scaled = Vector::<f32, V>::cast_from(values) * Vector::<f32, V>::cast_from(scale);
    Vector::<E, V>::cast_from(scaled)
}
