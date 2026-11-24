//! The decoding copy through a table or scales ([`Tile::lookup`], [`Tile::mul`]).

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::quant::scheme::QuantValue;

use crate::kind::Field;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

#[cube]
impl Tile {
    /// [`dequantize_from`](Tile::dequantize_from) a source in memory, read in the source's own
    /// loads ([`vector_tile`](Tile::vector_tile)), a run or a rectangle of its stored tiles. The
    /// source may carry extra dimensions only if each is one wide.
    pub(crate) fn copy_scaled_from<S: Numeric, R: Numeric>(&mut self, src: &Tile) {
        // It decodes in the unit that loads each line; an engine moving memory to memory lands the
        // bytes as they lie.
        let delivery = src.delivery();
        comptime!(assert!(
            delivery.route() == Route::Registers,
            "Tile::dequantize_from: a decoding copy cannot be delivered {delivery:?}; deliver it \
             SyncPerUnit, or stage it as it lies and decode out of the stage"
        ));
        let load = src.vector_tile();
        let sw = comptime!(load.values());
        let packing = src.packing();
        match comptime!(packing) {
            Packing::Plain | Packing::Words { .. } => {
                let size!(SW) = sw;
                self.copy_decoded::<S, R, R, SW>(
                    src,
                    src.nd_packed::<R, SW>(comptime!(Guard::Checked)),
                );
            }
            Packing::Packed { field: _ } => {
                let size!(WP) = comptime!(packing.physical(sw));
                self.copy_decoded::<S, R, u32, WP>(
                    src,
                    src.nd_words::<WP>(comptime!(Guard::Checked)),
                );
            }
        }
    }

    /// The copy, over `stored`: the source's loads as its buffer holds them, cyclically across
    /// the units that fill this tile.
    fn copy_decoded<O: Numeric, S: Numeric, I: Numeric, WP: Size>(
        &mut self,
        src: &Tile,
        stored: Masked<'_, Vector<I, WP>, CoordsDyn>,
    ) {
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        let vw = comptime!(lands.values());
        let size!(VW) = vw;
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let along = dst.dimension_at(dst.rank() - 1);
        let varies = src.factor.varies_along(comptime!(along));
        comptime!(check_scaled_copy(
            "Tile::dequantize_from",
            &space,
            &dst,
            &load,
            vw,
            varies
        ));
        // A window its holder shares is the holder's units' to fill, as a plain copy's is.
        let holder = comptime!(self.place.holder());
        let fill = self.copy_level();
        let fill = comptime!(fill.held_by(holder));
        let loads = comptime!(load.counts(&space).iter().product::<usize>());
        let mut out = self.nd_mut::<O, VW>();
        // The loads dealt to the units that fill, as a straight fill's lines are.
        for load in Walk::copy(&fill, comptime!(loads).runtime()) {
            src.decode_load_into::<S, O, I, WP, VW>(
                &stored,
                &mut out,
                load.coord(Level::COPIED) as u32,
                &dst,
                lands.clone(),
            );
        }
    }

    /// This source's `line`-th load, read out of `stored`, decoded under its table and scales
    /// into `out`, a window spanning `dst` written in `lands`.
    fn decode_load_into<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
        &self,
        stored: &Masked<'_, Vector<I, WP>, CoordsDyn>,
        out: &mut MaskedMut<'_, Vector<O, VW>, CoordsDyn>,
        line: u32,
        #[comptime] dst: &Space,
        #[comptime] lands: VectorTile,
    ) {
        let load = self.vector_tile();
        let (sw, vw) = comptime!((load.values(), lands.values()));
        let stored_as = self.packing();
        let space = comptime!(self.place.space.clone());
        let rank = space.rank();
        let mem = self.mem("Tile::dequantize_from");
        let start = load.start(line, &space);
        let held = stored.read(load.index(&start, &space));
        let powers_of_two = self.factor.powers_of_two();
        // The destination lines this load holds, `offset` their place in it, and where they land.
        #[unroll]
        for c in 0..(sw / vw) {
            let offset = c * vw;
            let mut at = Coords::<u32>::new();
            let mut to = Coords::<u32>::new();
            #[unroll]
            for p in 0..rank {
                let inner = comptime!(load.offset_along(offset, space.dimension_at(p)) as u32);
                let coord = start.at(p) + inner;
                at.push(coord);
                if dst.contains(space.dimension_at(p)) {
                    to.push(coord);
                }
            }
            let scale = self
                .factor
                .at_coords(&at, space.clone(), comptime!(Guard::Checked));
            write_decoded_line::<S, O, I, WP, VW>(
                held,
                comptime!(DecodedLine {
                    offset,
                    width: vw,
                    stored_as,
                    powers_of_two,
                }),
                mem,
                scale,
                out,
                lands.index(&to, dst),
            );
        }
    }
}

/// Where a line lies in its load and how its values are stored: what a decode of it reads at
/// compile time.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DecodedLine {
    /// Its first value's place in the load.
    offset: usize,
    /// Its values, a destination line's.
    width: usize,
    /// How the load's values are stored.
    stored_as: Packing,
    /// Whether every scale over the values is a power of two ([`Factor::powers_of_two`]).
    powers_of_two: bool,
}

impl DecodedLine {
    /// Whether the line is `e2m1` codes the fill turns into `e4m3` through their bits
    /// ([`e2m1_word_into_e4m3`]): codes read from their own bits, not the device's conversion,
    /// under powers of two alone, in whole words of the load and of the line, `into_e4m3` the
    /// destination's element, and no table.
    fn e2m1_into_e4m3(self, into_e4m3: bool, looks_up: bool) -> bool {
        let read_from_bits = matches!(
            self.stored_as,
            Packing::Packed {
                field: Field::Quant(QuantValue::E2M1)
                    | Field::Placed {
                        value: QuantValue::E2M1,
                        ..
                    }
            }
        );
        read_from_bits
            && self.powers_of_two
            && into_e4m3
            && !looks_up
            && self.width.is_multiple_of(E2M1_CODES_PER_WORD)
            && self.offset.is_multiple_of(E2M1_CODES_PER_WORD)
    }
}

/// The `VW` values of `held` that `line` names, under `scale`, written into `out` at `index` as
/// `O`: `e2m1` codes into `e4m3` through their bits where `line` allows it
/// ([`write_e2m1_line_into_e4m3`]), every other line through the general decode.
#[cube]
fn write_decoded_line<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
    held: Vector<I, WP>,
    #[comptime] line: DecodedLine,
    mem: &Memory,
    scale: f32,
    out: &mut MaskedMut<'_, Vector<O, VW>, CoordsDyn>,
    index: CoordsDyn,
) {
    let into_e4m3 = is_e4m3::<O>();
    let through_bits = comptime!(line.e2m1_into_e4m3(into_e4m3, mem.codebook.present()));
    if through_bits {
        write_e2m1_line_into_e4m3::<S, O, I, WP, VW>(held, line, mem, scale, out, index);
    } else {
        out.write(
            index,
            decode_line::<S, O, I, WP, VW>(held, line, mem, scale),
        );
    }
}

/// `e2m1` codes into `e4m3` under `scale`, a power of two: their bytes made of the codes' bits
/// ([`e2m1_word_into_e4m3`]) where every code is a normal `e4m3` value under it, exactly; the
/// general decode under any other.
#[cube]
fn write_e2m1_line_into_e4m3<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
    held: Vector<I, WP>,
    #[comptime] line: DecodedLine,
    mem: &Memory,
    scale: f32,
    out: &mut MaskedMut<'_, Vector<O, VW>, CoordsDyn>,
    index: CoordsDyn,
) {
    let scale_exponent =
        i32::cast_from(u32::reinterpret(scale) >> F32_STORED_MANTISSA_BITS) - F32_EXPONENT_BIAS;
    // Counted from the lowest as unsigned, an exponent below it wraps past the span.
    let normal = u32::cast_from(scale_exponent - E2M1_LOWEST_NORMAL_EXPONENT)
        <= (E2M1_HIGHEST_NORMAL_EXPONENT - E2M1_LOWEST_NORMAL_EXPONENT) as u32;
    if normal {
        out.write(
            index.clone(),
            e2m1_line_as_e4m3::<I, O, WP, VW>(held, line, scale_exponent),
        );
    } else {
        out.write(
            index,
            decode_line::<S, O, I, WP, VW>(held, line, mem, scale),
        );
    }
}

/// The `e2m1` codes of `held` that `line` names, eight a word, as their `e4m3` bytes under
/// `2^scale_exponent` ([`e2m1_word_into_e4m3`]).
#[cube]
fn e2m1_line_as_e4m3<I: Numeric, O: Numeric, WP: Size, VW: Size>(
    held: Vector<I, WP>,
    #[comptime] line: DecodedLine,
    scale_exponent: i32,
) -> Vector<O, VW> {
    let (offset, width) = comptime!((line.offset, line.width));
    let size!(VB) = comptime!(width / E4M3_PER_WORD);
    let mut bytes = Vector::<u32, VB>::empty();
    #[unroll]
    for w in 0..comptime!(width / E2M1_CODES_PER_WORD) {
        let word = u32::cast_from(held.extract(comptime!(offset / E2M1_CODES_PER_WORD + w)));
        let (low, high) = e2m1_word_into_e4m3(word, scale_exponent);
        bytes.insert(2 * w, low);
        bytes.insert(2 * w + 1, high);
    }
    Vector::<O, VW>::reinterpret(bytes)
}

/// The decoding copy split around a contraction, as the straight transport's register prefetch
/// splits a plain one ([`Stages::prefetched`]): each unit reads its loads of the packed source and
/// the scale of every line they land in into registers before the contraction, and decodes and
/// writes them into the stage after it.
#[cube]
impl Tile {
    /// Loads of `src` the decoding fill of this stage deals one unit, where a register fetch can
    /// hold them: a packed source read by the units, its loads landing whole destination lines,
    /// dealt to a known count of units. `None` elsewhere.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetches_in_registers`.
    pub(crate) fn decoded_fetch_turns(&self, src: &Tile) -> comptime_type!(Option<usize>) {
        let packing = src.packing();
        let delivery = src.delivery();
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        let holder = comptime!(self.place.holder());
        let fill = self.copy_level();
        let stage = comptime!(self.place.space.clone());
        // The fetch holds each block scale's word as stored, so the block scales are packed words
        // and the only level of scales that varies within a region.
        let as_words = src.factor.fetches_as_words();
        comptime!({
            // Counted off the stage: the walk asks before it windows the source to a region.
            let cells = stage.cells();
            let loads = cells / load.values();
            let fetchable = matches!(packing, Packing::Packed { .. })
                && as_words
                && delivery.route() == Route::Registers
                && cells.is_multiple_of(load.values())
                && load.values().is_multiple_of(lands.values());
            match fetchable {
                true => fill.held_by(holder).copy_turns(loads),
                false => None,
            }
        })
    }

    /// 32-bit registers one unit holds between [`fetch_decoded_from`](Tile::fetch_decoded_from)
    /// and [`store_decoded`](Tile::store_decoded) ([`Prefetch::decoded_registers`]). `None` where
    /// [`decoded_fetch_turns`](Tile::decoded_fetch_turns) is.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetches_in_registers`.
    pub(crate) fn decoded_fetched_registers(&self, src: &Tile) -> comptime_type!(Option<usize>) {
        let turns = self.decoded_fetch_turns(src);
        let packing = src.packing();
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        comptime!(turns.map(|turns| {
            let bits = match packing {
                Packing::Packed { field } => field.size_bits(),
                Packing::Plain | Packing::Words { .. } => {
                    unreachable!("Tile::decoded_fetched_registers: a decoding fetch is packed")
                }
            };
            Prefetch::decoded_registers(turns, load.values(), bits, lands.values())
        }))
    }

    /// Words a unit's fetched load is held in: the packed source's load as its buffer holds it.
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetch_buffers`.
    pub(crate) fn decoded_fetch_width(&self, src: &Tile) -> comptime_type!(usize) {
        let packing = src.packing();
        let load = src.vector_tile();
        comptime!(packing.physical(load.values()))
    }

    /// Scales a unit's fetched load lands its lines under: one a destination line.
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetch_buffers`.
    pub(crate) fn decoded_scales_per_load(&self, src: &Tile) -> comptime_type!(usize) {
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        comptime!(load.values() / lands.values())
    }

    /// Read this unit's loads of `src` into `raw`, as stored, and the scale of every line they land
    /// in into `scales`, [`decoded_scales_per_load`](Tile::decoded_scales_per_load) a load. Only
    /// where [`decoded_fetch_turns`](Tile::decoded_fetch_turns) holds; `I` is bound to `u32`.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetch_decoded_from<I: Numeric, WP: Size>(
        &self,
        src: &Tile,
        raw: &mut Array<Vector<I, WP>>,
        scales: &mut Array<u32>,
    ) {
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        let (sw, vw) = comptime!((load.values(), lands.values()));
        let per_load = sw / vw;
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let along = dst.dimension_at(dst.rank() - 1);
        let varies = src.factor.varies_along(comptime!(along));
        comptime!(check_scaled_copy(
            "Tile::fetch_decoded_from",
            &space,
            &dst,
            &load,
            vw,
            varies
        ));
        let rank = space.rank();
        let inner = src.factor.innermost_scale();
        let turns = self.decoded_fetch_turns(src);
        let turns = comptime!(turns.expect(
            "Tile::fetch_decoded_from: a decoding fill fetched into registers has a known share"
        ));
        let holder = comptime!(self.place.holder());
        let fill = self.copy_level();
        let fill = comptime!(fill.held_by(holder));
        let loads = comptime!(load.counts(&space).iter().product::<usize>());
        comptime!(assert_eq!(
            loads * sw,
            dst.cells(),
            "Tile::fetch_decoded_from: the region's loads cover the stage"
        ));
        let lines = Walk::copy(&fill, comptime!(loads).runtime());
        let stored = src.nd_words_as::<I, WP>(comptime!(Guard::Checked));
        // Each turn's load is held in a register of its own, so every turn is written out.
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let line = lines.region(t).coord(Level::COPIED) as u32;
                let start = load.start(line, &space);
                raw[t] = stored.read(load.index(&start, &space));
                #[unroll]
                for c in 0..per_load {
                    let offset = c * vw;
                    let mut at = Coords::<u32>::new();
                    #[unroll]
                    for p in 0..rank {
                        let inner =
                            comptime!(load.offset_along(offset, space.dimension_at(p)) as u32);
                        at.push(start.at(p) + inner);
                    }
                    // The scale's word as stored, nothing computed on it: a wait for the load
                    // here would be a wait for every load issued before it.
                    let own = inner.scale_coords(&at, &space);
                    scales[t * per_load + c] = inner.scale_word_at(&own, comptime!(Guard::Checked));
                }
            }
        }
    }

    /// Decode what [`fetch_decoded_from`](Tile::fetch_decoded_from) read of `src` and write it into
    /// this stage. The caller owns the rendezvous, as for
    /// [`store_fetched`](Tile::store_fetched).
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn store_decoded<I: Numeric, WP: Size>(
        &mut self,
        src: &Tile,
        raw: &Array<Vector<I, WP>>,
        scales: &Array<u32>,
    ) {
        let define!(Served) = self.element;
        let define!(Source) = src.read_element();
        let lands = self.vector_tile();
        let size!(VW) = comptime!(lands.values());
        self.store_decoded_lines::<Source, Served, I, WP, VW>(src, raw, scales);
    }

    /// [`store_decoded`](Tile::store_decoded), its lines `VW` wide, its source's values read as
    /// `S` and written as `O`.
    fn store_decoded_lines<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
        &mut self,
        src: &Tile,
        raw: &Array<Vector<I, WP>>,
        scales: &Array<u32>,
    ) {
        let (load, lands) = (src.vector_tile(), self.vector_tile());
        let (sw, vw) = comptime!((load.values(), lands.values()));
        let per_load = sw / vw;
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let rank = space.rank();
        let stored_as = src.packing();
        let packing = comptime!(stored_as.placed());
        let lift = comptime!(packing.lift() as f32);
        // Every scale outside the block scales, one value a region, as `Factor::at_coords`
        // multiplies them in after the innermost.
        let inner = src.factor.innermost_scale();
        let coarse = src.factor.coarse();
        let turns = self.decoded_fetch_turns(src);
        let turns = comptime!(turns.expect(
            "Tile::store_decoded: a decoding fill fetched into registers has a known share"
        ));
        let holder = comptime!(self.place.holder());
        let fill = self.copy_level();
        let fill = comptime!(fill.held_by(holder));
        let loads = comptime!(load.counts(&space).iter().product::<usize>());
        let lines = Walk::copy(&fill, comptime!(loads).runtime());
        let mem = src.mem("Tile::store_decoded");
        let mut out = self.nd_mut::<O, VW>();
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let line = lines.region(t).coord(Level::COPIED) as u32;
                let start = load.start(line, &space);
                let held = raw[t];
                #[unroll]
                for c in 0..per_load {
                    let offset = c * vw;
                    let mut at = Coords::<u32>::new();
                    let mut to = Coords::<u32>::new();
                    #[unroll]
                    for p in 0..rank {
                        let step =
                            comptime!(load.offset_along(offset, space.dimension_at(p)) as u32);
                        let coord = start.at(p) + step;
                        at.push(coord);
                        if dst.contains(space.dimension_at(p)) {
                            to.push(coord);
                        }
                    }
                    let own = inner.scale_coords(&at, &space);
                    let block = inner.scale_in_word::<f32>(scales[t * per_load + c], &own);
                    let scale = block * coarse * lift;
                    let values = values_at::<S, I, WP, VW>(held, offset, packing);
                    // Multiplied in `f32`, rounded to `O` once, as the copy it splits does.
                    let scaled = Vector::<f32, VW>::cast_from(mem.codebook.entries(values))
                        * Vector::<f32, VW>::cast_from(scale);
                    out.write(lands.index(&to, &dst), Vector::<O, VW>::cast_from(scaled));
                }
            }
        }
    }
}

/// What a copy through scales rests on, `reader` the copy: the packed side spans `src`, read or
/// written in `load`s, and the float side `dst`, written or read `vw` wide. `varies` is whether
/// the scales vary along the float side's innermost dimension.
pub(super) fn check_scaled_copy(
    reader: &str,
    src: &Space,
    dst: &Space,
    load: &VectorTile,
    vw: usize,
    varies: bool,
) {
    assert!(
        src.dimensions().all(|dimension| !src.is_dynamic(dimension)),
        "{reader}: a copy through scales walks the packed window through its own dimensions, which \
         needs their extents at expansion; this window spans a dynamic dimension. Copy a region a \
         level cuts to (a stage, a leaf) rather than the whole dynamic tile"
    );
    assert!(
        src.narrows_to(dst),
        "{reader}: a copy through scales takes the box it gives, and {src:?} does not narrow to \
         {dst:?} by dropping one-wide dimensions"
    );
    // A float line is `vw` consecutive values along its innermost dimension, so it sits inside one
    // load only where that is the load's finest dimension and its extent there holds whole lines.
    let along = dst.dimension_at(dst.rank() - 1);
    let &(finest, run) = load
        .extents()
        .first()
        .expect("a load holds at least one value");
    assert!(
        vw == 1 || (finest == along && run.is_multiple_of(vw)),
        "{reader}: a copy through scales takes each float line out of one load, and a {vw}-wide \
         line along {along:?} does not sit inside a load of {:?}",
        load.extents()
    );
    // One scale a float line, at its first value.
    assert!(
        vw == 1 || !varies,
        "{reader}: a float line runs {vw} values along {along:?}, which the scales vary along, so \
         one line lies under several scales; take the float side one value a line, or omit \
         {along:?} from the scales"
    );
}

/// The `VW` values of `held` that `line` names, decoded and multiplied by `scale` in `f32`,
/// rounded to `O` once. Read placed: an `e2m1` value decoded without its lift, which the scale
/// carries, one multiply a line rather than one a value; the lift times a scale is past a half's
/// range where its product with the value is not.
#[cube]
fn decode_line<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
    held: Vector<I, WP>,
    #[comptime] line: DecodedLine,
    mem: &Memory,
    scale: f32,
) -> Vector<O, VW> {
    let packing = line.stored_as.placed();
    let lift = comptime!(packing.lift() as f32);
    let values = values_at::<S, I, WP, VW>(held, line.offset, packing);
    let scaled = Vector::<f32, VW>::cast_from(mem.codebook.entries(values))
        * Vector::<f32, VW>::cast_from(scale * lift);
    Vector::<O, VW>::cast_from(scaled)
}

// What the decode of `e2m1` codes into `e4m3` through their bits reads of the two formats and of
// an `f32` scale.
/// `e2m1` codes a packed word holds.
const E2M1_CODES_PER_WORD: usize = 8;
/// `e4m3` values, a byte each, a word holds.
const E4M3_PER_WORD: usize = 4;
/// The mantissa bits an `f32` stores, below its exponent field.
const F32_STORED_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
/// The bias of an `f32`'s exponent field.
const F32_EXPONENT_BIAS: i32 = f32::MAX_EXP - 1;
/// The exponents of the powers of two under which every `e2m1` code is a normal `e4m3` value:
/// its `0.5` at `e4m3`'s smallest normal exponent, `2^-6`, at the lowest, its `6` at `448`'s
/// exponent, `2^8`, at the highest.
const E2M1_LOWEST_NORMAL_EXPONENT: i32 = -5;
const E2M1_HIGHEST_NORMAL_EXPONENT: i32 = 6;
/// The `e4m3` exponent field of `e2m1`'s smallest nonzero, `0.5`, under a scale of one: `2^-1`
/// at `e4m3`'s bias of seven.
const E2M1_HALF_EXPONENT_FIELD: i32 = 6;
/// Where an `e4m3` byte's exponent field starts.
const E4M3_EXPONENT_SHIFT: u32 = 3;

/// The eight `e2m1` codes of `word`, the low nibble first, as the `e4m3` bytes of their values
/// times `2^scale_exponent`, codes 0 to 3 in the first word and 4 to 7 in the second, for
/// `scale_exponent` from [`E2M1_LOWEST_NORMAL_EXPONENT`] to [`E2M1_HIGHEST_NORMAL_EXPONENT`]. A
/// code's magnitude bits `e1 e0 m` placed two bits up are its `e4m3` byte at exponent `e`, which
/// the exponent field of its `0.5` raises to the value's; `0.5`, `e2m1`'s one subnormal, is that
/// field alone, and zero stays zero.
#[cube(harness)]
fn e2m1_word_into_e4m3(word: u32, scale_exponent: i32) -> (u32, u32) {
    let exponent_field =
        u32::cast_from(scale_exponent + E2M1_HALF_EXPONENT_FIELD) << E4M3_EXPONENT_SHIFT;
    (
        e2m1_codes_into_e4m3(word & 0xFFFFu32, exponent_field),
        e2m1_codes_into_e4m3(word >> 16u32, exponent_field),
    )
}

/// The four `e2m1` codes in the low half of `codes` as `e4m3` bytes, `exponent_field` the field a
/// code's `0.5` takes, in place ([`e2m1_word_into_e4m3`]).
#[cube]
fn e2m1_codes_into_e4m3(codes: u32, exponent_field: u32) -> u32 {
    // The four codes a byte each, in the order of `k`.
    let spread = (codes | (codes << 8u32)) & 0x00FF_00FFu32;
    let spread = (spread | (spread << 4u32)) & 0x0F0F_0F0Fu32;
    let placed = (spread & 0x0707_0707u32) << 2u32;
    let sign = (spread << 4u32) & 0x8080_8080u32;
    // A byte past zero, and a byte that is `0.5`'s, whose mantissa bit the exponent takes.
    let nonzero = ((placed + 0x7C7C_7C7Cu32) & 0x8080_8080u32) >> 7u32;
    let not_half = ((placed ^ 0x0404_0404u32) + 0x7F7F_7F7Fu32) & 0x8080_8080u32;
    let half = ((not_half ^ 0x8080_8080u32) >> 7u32) << 2u32;
    (placed + nonzero * exponent_field - half) | sign
}

/// Whether `O` is `e4m3`.
#[allow(clippy::extra_unused_type_parameters)] // Read by the expansion.
#[cube]
fn is_e4m3<O: Numeric>() -> comptime_type!(bool) {
    intrinsic!(|scope| O::elem_type(scope) == ElemType::Float(FloatKind::E4M3))
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate cubecl as cubecl_runtime;
    use cubecl_common::e4m3;
    use cubecl_runtime::TestRuntime;

    /// Every `e2m1` code, the sixteen of two words, under every power of two from
    /// [`E2M1_LOWEST_NORMAL_EXPONENT`] to [`E2M1_HIGHEST_NORMAL_EXPONENT`]: each byte is the
    /// `e4m3` of the code's value times the scale, exactly, in the order of the codes.
    #[test]
    fn every_e2m1_code_becomes_its_exact_e4m3_byte_under_every_normal_scale() {
        const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let words = [0x7654_3210u32, 0xFEDC_BA98u32];
        for exponent in E2M1_LOWEST_NORMAL_EXPONENT..=E2M1_HIGHEST_NORMAL_EXPONENT {
            for (w, &word) in words.iter().enumerate() {
                let (low, high) = e2m1_word_into_e4m3::harness::<TestRuntime>(word, exponent);
                let got: Vec<u8> = [low, high].iter().flat_map(|b| b.to_le_bytes()).collect();
                let want: Vec<u8> = (0..8)
                    .map(|code| {
                        let sign = if w == 1 { -1.0 } else { 1.0 };
                        e4m3::from_f32(sign * MAGNITUDES[code] * 2f32.powi(exponent)).to_bits()
                    })
                    .collect();
                assert_eq!(got, want, "word {word:#010x} under 2^{exponent}");
            }
        }
    }
}
