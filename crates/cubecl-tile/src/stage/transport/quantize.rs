//! The encoding copy: a float tile written packed, under the block scales it computes
//! ([`Tile::quantize_from`]), the mirror of the decoding copy ([`Tile::dequantize_from`]).

use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantValue, ScaleDtype};
use cubecl_std::quant::round::round_up_to_dtype;
use cubecl_std::tensor::layout::CoordsDyn;

use super::scaled::check_scaled_copy;
use crate::*;

#[cube]
impl Tile {
    /// [`quantize_from`](Tile::quantize_from) a float source in memory, written in this tile's
    /// own loads ([`vector_tile`](Tile::vector_tile)), each read as whole lines of the source.
    /// The source may leave out an axis of the values only if it is one wide.
    ///
    /// The block scales are calibrated first ([`Factor::calibrate_from`]), and the window's
    /// holder meets before its units encode the values under them. Where one load is one block,
    /// alone under its scale, each load calibrates its block as it encodes it, with no meeting;
    /// where one load of the scales holds the scales of several such loads, a unit takes those
    /// loads together, calibrates each from its own lines and writes their scales as one load
    /// ([`blocks_a_scale_load`]).
    pub(crate) fn copy_quantized_from<F: Float>(&mut self, src: &Tile) {
        let coarse = self.factor.coarse();
        let packing = self.packing();
        let value = comptime!(packing.quantized_value(self.element));
        let scales = self.factor.innermost_scale();
        let space = comptime!(self.place.space.clone());
        let (load, scale_load) = (self.vector_tile(), scales.vector_tile());
        let fused = comptime!(
            one_block_a_load(&load, &space, &scales.place.space, &scale_load)
                || blocks_a_scale_load(&load, &space, &scales.place.space, &scale_load).is_some()
        );
        if !fused {
            self.factor.calibrate_from::<F>(src, coarse, value);
            // The scales are read back from wherever they are stored, global memory included.
            match comptime!(self.place.holder()) {
                ComputeScope::Plane => sync_plane_storage(),
                _ => sync_storage(),
            }
        }
        let sw = comptime!(load.values());
        // The written view borrows its own handle on this tile, which the copy still reads.
        let mut into = self.clone();
        match comptime!(packing) {
            Packing::Plain | Packing::Words { .. } => {
                let define!(Stored) = self.element;
                let size!(SW) = sw;
                self.copy_encoded::<F, Stored, SW>(src, into.nd_mut::<Stored, SW>(), coarse);
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = comptime!(packing.physical(sw));
                self.copy_encoded::<F, u32, WP>(src, into.nd_words_mut::<WP>(), coarse);
            }
        }
    }

    /// The copy, into `out`: this tile's loads as its buffer holds them, cyclically across the
    /// units that fill it.
    fn copy_encoded<F: Float, I: Numeric, WP: Size>(
        &self,
        src: &Tile,
        out: MaskedMut<'_, Vector<I, WP>, CoordsDyn>,
        coarse: f32,
    ) {
        let (load, line) = (self.vector_tile(), src.vector_tile());
        let lw = comptime!(line.values());
        let size!(LW) = lw;
        let space = comptime!(self.place.space.clone());
        let from = comptime!(src.place.space.clone());
        let along = from.axis_at(from.rank() - 1);
        let varies = self.factor.varies_along(comptime!(along));
        comptime!(check_scaled_copy(
            "Tile::quantize_from",
            &space,
            &from,
            &load,
            lw,
            varies
        ));
        let read = src.nd_packed::<F, LW>(comptime!(Guard::Checked));
        // A window its holder shares is the holder's units' to fill, as a plain copy's is.
        let holder = comptime!(self.place.holder());
        let fill = self.fill_units();
        let fill = comptime!(fill.held_by(holder));
        let scales = self.factor.innermost_scale();
        let scale_load = scales.vector_tile();
        let grouped = comptime!(blocks_a_scale_load(
            &load,
            &space,
            &scales.place.space,
            &scale_load
        ));
        match comptime!(grouped) {
            Some(_) => self.encode_groups_into::<F, I, WP, LW>(
                read,
                out,
                coarse,
                fill,
                comptime!(from.clone()),
                comptime!(line.clone()),
            ),
            None => self.encode_loads_into::<F, I, WP, LW>(
                read,
                out,
                coarse,
                fill,
                comptime!(from.clone()),
                comptime!(line.clone()),
            ),
        }
    }

    /// This tile's groups of loads encoded into `out` ([`encode_group_into`](Tile::encode_group_into)),
    /// cyclically across `fill`'s units, each unit a group at a time: what a tile whose loads
    /// group under one load of their scales takes ([`blocks_a_scale_load`]).
    fn encode_groups_into<F: Float, I: Numeric, WP: Size, LW: Size>(
        &self,
        read: Masked<'_, Vector<F, LW>, CoordsDyn>,
        mut out: MaskedMut<'_, Vector<I, WP>, CoordsDyn>,
        coarse: f32,
        #[comptime] fill: FillUnits,
        #[comptime] from: Space,
        #[comptime] line: VectorTile,
    ) {
        let load = self.vector_tile();
        let space = comptime!(self.place.space.clone());
        let n = self.loads_a_scale_load();
        let loads = comptime!(load.counts(&space).iter().product::<usize>());
        let first = FillUnits::worker(fill) as u32;
        let stride = FillUnits::workers(fill) as u32;
        for at_group in range_stepped(first, comptime!((loads / n) as u32), stride) {
            self.encode_group_into::<F, I, WP, LW>(
                &read,
                &mut out,
                at_group,
                coarse,
                comptime!(from.clone()),
                comptime!(line.clone()),
            );
        }
    }

    /// This tile's loads encoded into `out`, cyclically across `fill`'s units, each read out of
    /// `read`, a source spanning `from` read in `line`s
    /// ([`encode_load_into`](Tile::encode_load_into)).
    fn encode_loads_into<F: Float, I: Numeric, WP: Size, LW: Size>(
        &self,
        read: Masked<'_, Vector<F, LW>, CoordsDyn>,
        mut out: MaskedMut<'_, Vector<I, WP>, CoordsDyn>,
        coarse: f32,
        #[comptime] fill: FillUnits,
        #[comptime] from: Space,
        #[comptime] line: VectorTile,
    ) {
        let load = self.vector_tile();
        let space = comptime!(self.place.space.clone());
        let units = fill.count;
        let loads = comptime!(load.counts(&space).iter().product::<usize>());
        // A comptime worker count unrolls the loads, as a straight fill unrolls its lines: a
        // runtime stride stalls each store on Metal's in-order pipe.
        if units > 0 && loads.div_ceil(units) <= 8 {
            #[unroll]
            for t in 0..loads.div_ceil(units) {
                let at_load = FillUnits::worker(fill) as u32 + comptime!((t * units) as u32);
                if (t + 1) * units <= loads {
                    self.encode_load_into::<F, I, WP, LW>(
                        &read,
                        &mut out,
                        at_load,
                        coarse,
                        from.clone(),
                        comptime!(line.clone()),
                    );
                } else {
                    if at_load < comptime!(loads as u32) {
                        self.encode_load_into::<F, I, WP, LW>(
                            &read,
                            &mut out,
                            at_load,
                            coarse,
                            from.clone(),
                            comptime!(line.clone()),
                        );
                    }
                }
            }
        } else {
            let first = FillUnits::worker(fill) as u32;
            let stride = FillUnits::workers(fill) as u32;
            for at_load in range_stepped(first, comptime!(loads as u32), stride) {
                self.encode_load_into::<F, I, WP, LW>(
                    &read,
                    &mut out,
                    at_load,
                    coarse,
                    from.clone(),
                    comptime!(line.clone()),
                );
            }
        }
    }

    /// The loads one load of this tile's scales covers, each one block
    /// ([`blocks_a_scale_load`]): what a group of [`encode_group_into`](Tile::encode_group_into)
    /// holds.
    fn loads_a_scale_load(&self) -> comptime_type!(usize) {
        let scales = self.factor.innermost_scale();
        let (load, scale_load) = (self.vector_tile(), scales.vector_tile());
        let space = comptime!(self.place.space.clone());
        comptime!(
            blocks_a_scale_load(&load, &space, &scales.place.space, &scale_load)
                .expect("a group is taken where one load of the scales covers several loads")
        )
    }

    /// This tile's `at_group`-th group of loads, each one block, whose scales are one load
    /// of the scales ([`blocks_a_scale_load`]): every block's lines read out of `read`, a source
    /// spanning `from` read in `line`s, its scale calibrated from them, the `n` scales written as
    /// that one load, and each block's values encoded under its scale into `out`.
    fn encode_group_into<F: Float, I: Numeric, WP: Size, LW: Size>(
        &self,
        read: &Masked<'_, Vector<F, LW>, CoordsDyn>,
        out: &mut MaskedMut<'_, Vector<I, WP>, CoordsDyn>,
        at_group: u32,
        coarse: f32,
        #[comptime] from: Space,
        #[comptime] line: VectorTile,
    ) {
        let load = self.vector_tile();
        let n = self.loads_a_scale_load();
        let (sw, lw) = comptime!((load.values(), line.values()));
        let lines = comptime!(sw / lw);
        let packing = self.packing();
        let value = comptime!(packing.quantized_value(self.element));
        let two_levels = comptime!(self.factor.depth() > 1);
        let scales = self.factor.innermost_scale();
        let scale_packing = scales.packing();
        let dtype = comptime!(scale_packing.scale_dtype(scales.element));
        let space = comptime!(self.place.space.clone());
        let rank = comptime!(space.rank());
        let first = at_group * comptime!(n as u32);
        let size!(NB) = n;
        let mut held = Array::<Vector<F, LW>>::new(comptime!(n * lines));
        let mut inner = Vector::<f32, NB>::empty();
        #[unroll]
        for j in 0..n {
            let start = load.start(first + comptime!(j as u32), &space);
            let mut most = F::new(0.0);
            #[unroll]
            for c in 0..lines {
                let offset = comptime!(c * lw);
                let mut to = Coords::<u32>::new();
                #[unroll]
                for p in 0..rank {
                    if comptime!(from.contains(space.axis_at(p))) {
                        let inner = comptime!(load.offset_along(offset, space.axis_at(p)) as u32);
                        to.push(start.at(p) + inner);
                    }
                }
                let at = comptime!(j * lines + c);
                held[at] = read.read(line.index(&to, &from));
                most = max(most, largest::<F, LW>(held[at]));
            }
            inner.insert(
                j,
                f32::cast_from(block_scale::<F>(most, coarse, value, dtype, two_levels)),
            );
        }
        self.factor
            .write_at::<NB>(&load.start(first, &space), comptime!(space.clone()), inner);
        #[unroll]
        for j in 0..n {
            let start = load.start(first + comptime!(j as u32), &space);
            let scale = inner.extract(j);
            let effective = if comptime!(two_levels) {
                scale * coarse
            } else {
                scale
            };
            let mut words = Vector::<I, WP>::new(I::from_int(0));
            #[unroll]
            for c in 0..lines {
                let values =
                    encoded::<F, LW>(held[comptime!(j * lines + c)], effective, value, two_levels);
                words = place_at::<I, WP, LW>(words, values, comptime!(c * lw), packing);
            }
            out.write(load.index(&start, &space), words);
        }
    }

    /// This tile's `at_load`-th load, its lines read out of `read`, a source spanning `from` read
    /// in `line`s, encoded under their scales and written into `out`. Where the load is one block
    /// alone under its scale, the scale is calibrated here, from the load's lines, and written
    /// beside it; elsewhere it was, and is read back.
    fn encode_load_into<F: Float, I: Numeric, WP: Size, LW: Size>(
        &self,
        read: &Masked<'_, Vector<F, LW>, CoordsDyn>,
        out: &mut MaskedMut<'_, Vector<I, WP>, CoordsDyn>,
        at_load: u32,
        coarse: f32,
        #[comptime] from: Space,
        #[comptime] line: VectorTile,
    ) {
        let load = self.vector_tile();
        let (sw, lw) = comptime!((load.values(), line.values()));
        let lines = sw / lw;
        let packing = self.packing();
        let value = comptime!(packing.quantized_value(self.element));
        let two_levels = comptime!(self.factor.depth() > 1);
        let scales = self.factor.innermost_scale();
        let space = comptime!(self.place.space.clone());
        let scale_load = scales.vector_tile();
        let fused = comptime!(one_block_a_load(
            &load,
            &space,
            &scales.place.space,
            &scale_load
        ));
        let rank = space.rank();
        let start = load.start(at_load, &space);
        // The source lines this load holds, `offset` their place in it, where each is read, and
        // the scale it is over.
        let mut held = Array::<Vector<F, LW>>::new(lines);
        let mut over = Array::<f32>::new(lines);
        let mut most = F::new(0.0);
        #[unroll]
        for c in 0..lines {
            let offset = c * lw;
            let mut at = Coords::<u32>::new();
            let mut to = Coords::<u32>::new();
            #[unroll]
            for p in 0..rank {
                let inner = comptime!(load.offset_along(offset, space.axis_at(p)) as u32);
                let coord = start.at(p) + inner;
                at.push(coord);
                if from.contains(space.axis_at(p)) {
                    to.push(coord);
                }
            }
            held[c] = read.read(line.index(&to, &from));
            if comptime!(fused) {
                most = max(most, largest::<F, LW>(held[c]));
            } else {
                over[c] = self
                    .factor
                    .at_coords(&at, space.clone(), comptime!(Guard::Checked));
            }
        }
        if comptime!(fused) {
            let scale_packing = scales.packing();
            let dtype = comptime!(scale_packing.scale_dtype(scales.element));
            let inner = f32::cast_from(block_scale::<F>(most, coarse, value, dtype, two_levels));
            self.factor
                .write_at::<Const<1>>(&start, space.clone(), Vector::new(inner));
            let effective = if comptime!(two_levels) {
                inner * coarse
            } else {
                inner
            };
            #[unroll]
            for c in 0..lines {
                over[c] = effective;
            }
        }
        let mut words = Vector::<I, WP>::new(I::from_int(0));
        #[unroll]
        for c in 0..lines {
            let values = encoded::<F, LW>(held[c], over[c], value, two_levels);
            words = place_at::<I, WP, LW>(words, values, c * lw, packing);
        }
        out.write(load.index(&start, &space), words);
    }

    /// This tile's scales, the innermost of a factor, calibrated from `src`, the float values
    /// they scale, for values written as `value` ([`Factor::calibrate_from`]): its scales
    /// cyclically across the units that fill it, each block's largest magnitude taken over
    /// `src`'s lines. Where the units are lanes, they take a block each, and a word's scales meet
    /// in its first unit through the plane, which writes the word; the units take as many turns
    /// as the next, so each meeting is the whole plane's. Otherwise a unit takes a word at a time.
    /// `two_levels` is whether `coarse` sits over the scales.
    #[allow(dead_code)] // Reached through its expand, from [`Factor::calibrate_from`].
    pub(crate) fn calibrate<F: Float>(
        &mut self,
        src: &Tile,
        coarse: f32,
        #[comptime] value: QuantValue,
        #[comptime] two_levels: bool,
    ) {
        let (load, line) = (self.vector_tile(), src.vector_tile());
        let n = comptime!(load.values());
        let size!(N) = n;
        let size!(LW) = comptime!(line.values());
        let space = comptime!(self.place.space.clone());
        let from = comptime!(src.place.space.clone());
        let read = src.nd_packed::<F, LW>(comptime!(Guard::Checked));
        let holder = comptime!(self.place.holder());
        let fill = self.fill_units();
        let fill = comptime!(fill.held_by(holder));
        let blocks = comptime!((load.counts(&space).iter().product::<usize>() * n) as u32);
        // Lanes share a word's blocks where the plane is a word wide; a unit alone, or a plane
        // narrower than a word, takes a word at a time. The plane's width is the launch's, the
        // same for every unit, so the shuffles below are the whole plane's.
        let lanes = if comptime!(n > 1 && holder != ComputeScope::Unit) {
            CUBE_DIM_X >= comptime!(n as u32)
        } else {
            false.runtime()
        };
        let per = select(lanes, 1u32, comptime!(n as u32));
        let shares = blocks.runtime() / per;
        let unit = FillUnits::worker(fill) as u32;
        let width = FillUnits::workers(fill) as u32;
        for turn in 0..shares.div_ceil(width) {
            let at = turn * width + unit;
            let held = at < shares;
            // A unit past the scales calibrates the first blocks again, and writes nothing.
            let at = select(held, at, 0u32);
            let block = at * per;
            let start = load.start(block / comptime!(n as u32), &space);
            let c = block % comptime!(n as u32);
            let mut scales = Vector::<f32, N>::empty();
            #[unroll]
            for k in 0..n {
                if comptime!(k as u32) < per {
                    let scale = self.calibrate_block::<F, LW>(
                        &read,
                        &start,
                        c + comptime!(k as u32),
                        coarse,
                        from.clone(),
                        comptime!(line.clone()),
                        value,
                        two_levels,
                    );
                    scales.insert(k, scale);
                }
            }
            if lanes {
                let scale = scales.extract(0usize);
                let first = UNIT_POS_PLANE - c;
                #[unroll]
                for k in 0..n {
                    scales.insert(k, plane_shuffle(scale, first + comptime!(k as u32)));
                }
            }
            if held && c == 0 {
                self.write_scales::<N>(&start, scales);
            }
        }
    }

    /// The scale of the `c`-th block of this tile's load of scales starting at `start`,
    /// calibrated: the block read out of `read`, a source spanning `from` in `line`s, every value
    /// at its scale's coordinates along the axes the scales span and every position along the
    /// ones they omit.
    #[allow(clippy::too_many_arguments)]
    fn calibrate_block<F: Float, LW: Size>(
        &self,
        read: &Masked<'_, Vector<F, LW>, CoordsDyn>,
        start: &Coords<u32>,
        c: u32,
        coarse: f32,
        #[comptime] from: Space,
        #[comptime] line: VectorTile,
        #[comptime] value: QuantValue,
        #[comptime] two_levels: bool,
    ) -> f32 {
        let load = self.vector_tile();
        let packing = self.packing();
        let dtype = comptime!(packing.scale_dtype(self.element));
        let space = comptime!(self.place.space.clone());
        let (along, _) = comptime!(load.along_one_axis("Tile::quantize_from"));
        let block = from.subspace(&from.difference(&space));
        let lines = block.tiles(&line.space());
        let mut most = F::new(0.0);
        #[unroll]
        for l in 0..lines {
            let mut at = Coords::<u32>::new();
            #[unroll]
            for p in 0..from.rank() {
                let axis = from.axis_at(p);
                if space.contains(axis) {
                    let coord = start.at(space.position(axis));
                    if axis == along {
                        at.push(coord + c);
                    } else {
                        at.push(coord);
                    }
                } else {
                    at.push(comptime!(line.offset_in(l, &block, axis) as u32).runtime());
                }
            }
            most = max(most, largest::<F, LW>(read.read(line.index(&at, &from))));
        }
        f32::cast_from(block_scale::<F>(most, coarse, value, dtype, two_levels))
    }
}

/// The largest magnitude of `line`.
#[cube]
fn largest<F: Float, N: Size>(line: Vector<F, N>) -> F {
    let mut most = F::new(0.0);
    #[unroll]
    for e in 0..N::value() {
        most = max(most, F::abs(line.extract(e)));
    }
    most
}

/// A block's scale, its largest magnitude `most`, as burn calibrates it for values written as
/// `value`: the block's range, twice its largest, over the format's, in the source's element,
/// rounded up to `dtype`, which never clips the block's largest. Spelled as burn's ops: a
/// device's fast division rounds `2a / (b - a)` and `a / ((b - a) / 2)` apart. Under a coarser
/// scale, `coarse`, where `two_levels`, the block's is taken relative to it, in `f32`.
#[cube]
fn block_scale<F: Float>(
    most: F,
    coarse: f32,
    #[comptime] value: QuantValue,
    #[comptime] dtype: ScaleDtype,
    #[comptime] two_levels: bool,
) -> F {
    let (low, high) = value.range();
    let wanted = most * F::new(2.0) / F::new(comptime!((high - low) as f64));
    if comptime!(two_levels) {
        round_up_to_dtype::<F>(F::cast_from(f32::cast_from(wanted) / coarse), dtype)
    } else {
        round_up_to_dtype::<F>(wanted, dtype)
    }
}

/// `line` over `scale`, its effective scale, on `value`'s grid: a float format clamped to its
/// range, an integer one rounded to the nearest integer and clamped. Divided in the source's
/// element, as burn divides, or in `f32` under a coarser scale (`two_levels`). An all-zero
/// block's scale is zero, which no value is over: its values are zero.
#[cube]
fn encoded<F: Float, N: Size>(
    line: Vector<F, N>,
    scale: f32,
    #[comptime] value: QuantValue,
    #[comptime] two_levels: bool,
) -> Vector<f32, N> {
    let (scaled, zero) = if comptime!(two_levels) {
        (
            Vector::<f32, N>::cast_from(line) / Vector::new(scale),
            scale == 0.0,
        )
    } else {
        let scale = F::cast_from(scale);
        (
            Vector::<f32, N>::cast_from(line / Vector::new(scale)),
            scale == F::new(0.0),
        )
    };
    let (low, high) = value.range();
    let on_grid = match comptime!(value) {
        QuantValue::E2M1 | QuantValue::E4M3 | QuantValue::E5M2 => {
            clamp(scaled, Vector::new(low), Vector::new(high))
        }
        _ => clamp(Vector::round(scaled), Vector::new(low), Vector::new(high)),
    };
    select(zero, Vector::new(0.0), on_grid)
}

/// Whether each `load` of values spanning `values` is one block alone under its scale, the
/// scales spanning `scales` in `scale_load`s: then a load's own lines calibrate its scale.
fn one_block_a_load(
    load: &VectorTile,
    values: &Space,
    scales: &Space,
    scale_load: &VectorTile,
) -> bool {
    let block = values.subspace(&values.difference(scales));
    scale_load.values() == 1
        && load.values() == block.cells()
        && load
            .extents()
            .iter()
            .all(|&(axis, extent)| block.contains(axis) && block.extent(axis) == extent)
}

/// The loads one `scale_load` of the scales spanning `scales` covers, where each `load` of values
/// spanning `values` is one block and a load of the scales holds the scales of `n` such blocks in
/// a row, along the axis the loads step through first: `Some(n)` for `n` past one. A unit taking
/// those `n` loads together calibrates every block from its own lines and writes their scales as
/// that one load, which no other unit shares.
fn blocks_a_scale_load(
    load: &VectorTile,
    values: &Space,
    scales: &Space,
    scale_load: &VectorTile,
) -> Option<usize> {
    let block = values.subspace(&values.difference(scales));
    let one_block = load.values() == block.cells()
        && load
            .extents()
            .iter()
            .all(|&(axis, extent)| block.contains(axis) && block.extent(axis) == extent);
    let n = scale_load.values();
    let [(axis, extent)] = scale_load.extents() else {
        return None;
    };
    if !one_block || n <= 1 || *extent != n {
        return None;
    }
    // The loads step `axis` first: every axis after it is one load long, and `n` divides it.
    let counts = load.counts(values);
    let position = values.position(*axis);
    (counts[position].is_multiple_of(n) && counts[position + 1..].iter().all(|&count| count == 1))
        .then_some(n)
}
