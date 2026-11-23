//! The decoding copy through a table or scales ([`Tile::lookup`], [`Tile::mul`]).

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::quant::scheme::QuantValue;

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
        // Read placed: an `e2m1` value decoded without its lift, which the scale carries, one
        // multiply a line rather than one a value.
        let stored_as = self.packing();
        let packing = comptime!(stored_as.placed());
        let lift = comptime!(packing.lift() as f32);
        let space = comptime!(self.place.space.clone());
        let rank = space.rank();
        let mem = self.mem("Tile::dequantize_from");
        let start = load.start(line, &space);
        let held = stored.read(load.index(&start, &space));
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
            // `e2m1` codes into `e4m3` under a power-of-two scale, exactly where every code stays
            // a normal `e4m3` one: their bytes made of the codes' bits; the general decode
            // otherwise, and for every other field and element.
            let into_e4m3 = is_e4m3::<O>();
            let e2m1_into_e4m3 = comptime!(
                is_e2m1(stored_as)
                    && into_e4m3
                    && vw.is_multiple_of(E2M1_PER_WORD)
                    && offset.is_multiple_of(E2M1_PER_WORD)
                    && !mem.codebook.present()
            );
            let unlifted = self
                .factor
                .at_coords(&at, space.clone(), comptime!(Guard::Checked));
            let steps =
                i32::cast_from(u32::reinterpret(unlifted) >> F32_MANTISSA_BITS) - F32_EXPONENT_BIAS;
            let in_range = steps >= E2M1_LOWEST_STEPS && steps <= E2M1_HIGHEST_STEPS;
            if comptime!(e2m1_into_e4m3) {
                if in_range {
                    let size!(VB) = comptime!(vw / BYTES_PER_WORD);
                    let mut bytes = Vector::<u32, VB>::empty();
                    #[unroll]
                    for w in 0..comptime!(vw / E2M1_PER_WORD) {
                        let word =
                            u32::cast_from(held.extract(comptime!(offset / E2M1_PER_WORD + w)));
                        let (low, high) = e2m1_word_into_e4m3(word, steps);
                        bytes.insert(comptime!(2 * w), low);
                        bytes.insert(comptime!(2 * w + 1), high);
                    }
                    out.write(lands.index(&to, dst), Vector::<O, VW>::reinterpret(bytes));
                } else {
                    let decoded =
                        decode_line::<S, O, I, WP, VW>(held, offset, packing, mem, unlifted * lift);
                    out.write(lands.index(&to, dst), decoded);
                }
            } else {
                let decoded =
                    decode_line::<S, O, I, WP, VW>(held, offset, packing, mem, unlifted * lift);
                out.write(lands.index(&to, dst), decoded);
            }
        }
    }
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

/// The `VW` values of `held` from `offset` decoded and multiplied by `scale` in `f32`, rounded to
/// `O` once: a placed `e2m1` value is a lift short, which `scale` carries, and the lift times a
/// scale is past a half's range where its product with the value is not.
#[cube]
fn decode_line<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
    held: Vector<I, WP>,
    #[comptime] offset: usize,
    #[comptime] packing: Packing,
    mem: &Memory,
    scale: f32,
) -> Vector<O, VW> {
    let values = values_at::<S, I, WP, VW>(held, offset, packing);
    let scaled = Vector::<f32, VW>::cast_from(mem.codebook.entries(values))
        * Vector::<f32, VW>::cast_from(scale);
    Vector::<O, VW>::cast_from(scaled)
}

/// `e2m1` codes a packed word holds.
const E2M1_PER_WORD: usize = 8;
/// Bytes, `e4m3` values, a word holds.
const BYTES_PER_WORD: usize = 4;
/// `f32`'s mantissa bits and its exponent's bias.
const F32_MANTISSA_BITS: u32 = 23;
const F32_EXPONENT_BIAS: i32 = 127;
/// The scales, as powers of two, under which every `e2m1` code is a normal `e4m3` one: its `0.5`
/// at `e4m3`'s smallest normal exponent at the lowest, its `6` at `448`'s at the highest.
const E2M1_LOWEST_STEPS: i32 = -5;
const E2M1_HIGHEST_STEPS: i32 = 6;

/// Whether `packing` holds `e2m1` codes, however its field reads them.
fn is_e2m1(packing: Packing) -> bool {
    match packing {
        Packing::Packed { field } => field.code() == Some(QuantValue::E2M1),
        Packing::Plain | Packing::Words { .. } => false,
    }
}

/// The eight `e2m1` codes of `word`, the low nibble first, as the `e4m3` bytes of their values
/// times `2^steps`, codes 0 to 3 in the first word and 4 to 7 in the second, for `steps` from
/// [`E2M1_LOWEST_STEPS`] to [`E2M1_HIGHEST_STEPS`]. A code's magnitude bits `e1 e0 m` placed two
/// bits up are its `e4m3` byte at exponent `e`, which the scale's steps, six more, raise to the
/// value's; `0.5`, `e2m1`'s one subnormal, is that byte's exponent alone, and zero stays zero.
#[cube]
pub(crate) fn e2m1_word_into_e4m3(word: u32, steps: i32) -> (u32, u32) {
    let raised = u32::cast_from(steps + E2M1_SUBNORMAL_EXPONENT) << 3u32;
    (
        e2m1_codes_into_e4m3(word & 0xFFFFu32, raised),
        e2m1_codes_into_e4m3(word >> 16u32, raised),
    )
}

/// The four `e2m1` codes in the low half of `codes` as `e4m3` bytes, `raised` the exponent field
/// a code's `0.5` takes, shifted into place ([`e2m1_word_into_e4m3`]).
#[cube]
fn e2m1_codes_into_e4m3(codes: u32, raised: u32) -> u32 {
    // The four codes a byte each, in the order of `k`.
    let spread = (codes | (codes << 8u32)) & 0x00FF_00FFu32;
    let spread = (spread | (spread << 4u32)) & 0x0F0F_0F0Fu32;
    let placed = (spread & 0x0707_0707u32) << 2u32;
    let sign = (spread << 4u32) & 0x8080_8080u32;
    // A byte past zero, and a byte that is `0.5`'s, whose mantissa bit the exponent takes.
    let nonzero = ((placed + 0x7C7C_7C7Cu32) & 0x8080_8080u32) >> 7u32;
    let not_half = ((placed ^ 0x0404_0404u32) + 0x7F7F_7F7Fu32) & 0x8080_8080u32;
    let half = ((not_half ^ 0x8080_8080u32) >> 7u32) << 2u32;
    (placed + nonzero * raised - half) | sign
}

/// Whether `O` is `e4m3`.
#[allow(clippy::extra_unused_type_parameters)] // Read by the expansion.
#[cube]
fn is_e4m3<O: Numeric>() -> comptime_type!(bool) {
    intrinsic!(|scope| O::elem_type(scope) == ElemType::Float(FloatKind::E4M3))
}

/// The `e4m3` exponent `e2m1`'s smallest nonzero, `0.5`, takes at a scale of one: `2^-1`.
const E2M1_SUBNORMAL_EXPONENT: i32 = 6;
