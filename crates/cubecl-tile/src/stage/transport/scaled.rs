//! The decoding copy through a table or scales ([`Tile::lookup`], [`Tile::mul`]).

use cubecl::e2m1;
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
        let mem = self.mem("Tile::dequantize_from");
        let start = load.start(line, &space);
        let held = stored.read(load.index(&start, &space));
        let through_products = reads_products::<O>(
            comptime!(LoadDecode {
                stored_as,
                loaded: sw,
                line: vw,
            }),
            mem,
        );
        let scales = self.scales_per_load(comptime!(lands.clone()));
        let lines_under = comptime!(sw / vw / scales);
        #[unroll]
        for s in 0..scales {
            let first = line_coords(&start, &load, comptime!(s * lines_under * vw), &space);
            let decode = ScaledDecode::under::<O>(
                self.factor
                    .at_coords(&first, space.clone(), comptime!(Guard::Checked)),
                through_products,
            );
            #[unroll]
            for c in 0..lines_under {
                let offset = comptime!((s * lines_under + c) * vw);
                let to = landing(&line_coords(&start, &load, offset, &space), &space, dst);
                out.write(
                    lands.index(&to, dst),
                    decode.line::<S, O, I, WP, VW>(held, offset, mem, stored_as),
                );
            }
        }
    }

    /// Scales a load of these values lands its lines, as `lands` holds them, under: one for the
    /// load where it lies under one, one a destination line otherwise.
    fn scales_per_load(&self, #[comptime] lands: VectorTile) -> comptime_type!(usize) {
        let load = self.vector_tile();
        let space = comptime!(self.place.space.clone());
        let shared = self.factor.broadcasts_over(&load, space);
        comptime!(match shared {
            true => 1,
            false => load.values() / lands.values(),
        })
    }
}

/// What a load's decode reads at compile time.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct LoadDecode {
    /// How the load's values are stored.
    stored_as: Packing,
    /// Its values.
    loaded: usize,
    /// The values of one destination line.
    line: usize,
}

impl LoadDecode {
    /// Whether the load's values are `e2m1` codes read from their own bits, not the device's
    /// conversion, in whole words of the load and of its lines.
    fn holds_e2m1_words(self) -> bool {
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
            && self.loaded.is_multiple_of(E2M1_CODES_PER_WORD)
            && self.line.is_multiple_of(E2M1_CODES_PER_WORD)
    }
}

/// Whether a load decodes into `O`, 8 bits a value, by looking each of its codes up in their
/// products with its scale ([`ScaledDecode`]): the codes index no table of their own.
#[cube]
fn reads_products<O: Numeric>(#[comptime] load: LoadDecode, mem: &Memory) -> comptime_type!(bool) {
    let into_fp8 = is_fp8::<O>();
    comptime!(into_fp8 && load.holds_e2m1_words() && !mem.codebook.present())
}

/// The values under one scale as their codes decode: the scale, and, for `e2m1` codes decoded into
/// an 8-bit float, the product of every magnitude a code holds with it, which a code's value is
/// looked up in rather than multiplied. Every value is the same whichever way it is made: its
/// product in `f32`, rounded once.
#[derive(CubeType, Clone)]
struct ScaledDecode {
    scale: f32,
    /// The products of the magnitudes of codes 0 to 3, a byte each; zero where the values are
    /// multiplied.
    products_low: u32,
    /// Those of codes 4 to 7.
    products_high: u32,
    #[cube(comptime)]
    through_products: bool,
}

#[cube]
impl ScaledDecode {
    fn under<O: Numeric>(scale: f32, #[comptime] through_products: bool) -> ScaledDecode {
        if through_products {
            let (products_low, products_high) = e2m1_products::<O>(scale);
            ScaledDecode {
                scale,
                products_low,
                products_high,
                through_products,
            }
        } else {
            ScaledDecode {
                scale,
                products_low: 0u32.runtime(),
                products_high: 0u32.runtime(),
                through_products,
            }
        }
    }

    /// The `VW` values of `held` from `offset` on.
    fn line<S: Numeric, O: Numeric, I: Numeric, WP: Size, VW: Size>(
        &self,
        held: Vector<I, WP>,
        #[comptime] offset: usize,
        mem: &Memory,
        #[comptime] stored_as: Packing,
    ) -> Vector<O, VW> {
        if comptime!(self.through_products) {
            self.looked_up::<O, I, WP, VW>(held, offset)
        } else {
            decode_line::<S, O, I, WP, VW>(held, offset, mem, stored_as, self.scale)
        }
    }

    /// The `VW` values of `e2m1` codes of `held` from `offset` on, eight a word.
    fn looked_up<O: Numeric, I: Numeric, WP: Size, VW: Size>(
        &self,
        held: Vector<I, WP>,
        #[comptime] offset: usize,
    ) -> Vector<O, VW> {
        let width = VW::value();
        let size!(VB) = comptime!(width / FP8_PER_WORD);
        let mut bytes = Vector::<u32, VB>::empty();
        #[unroll]
        for w in 0..comptime!(width / E2M1_CODES_PER_WORD) {
            let word = u32::cast_from(held.extract(comptime!(offset / E2M1_CODES_PER_WORD + w)));
            let (low, high) = e2m1_word_looked_up(word, self.products_low, self.products_high);
            bytes.insert(2 * w, low);
            bytes.insert(2 * w + 1, high);
        }
        Vector::<O, VW>::reinterpret(bytes)
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
        let load = src.vector_tile();
        let per_load = self.decoded_scales_per_load(src);
        comptime!(turns.map(|turns| {
            let bits = match packing {
                Packing::Packed { field } => field.size_bits(),
                Packing::Plain | Packing::Words { .. } => {
                    unreachable!("Tile::decoded_fetched_registers: a decoding fetch is packed")
                }
            };
            Prefetch::decoded_registers(turns, load.values(), bits, load.values() / per_load)
        }))
    }

    /// Words a unit's fetched load is held in: the packed source's load as its buffer holds it.
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetch_buffers`.
    pub(crate) fn decoded_fetch_width(&self, src: &Tile) -> comptime_type!(usize) {
        let packing = src.packing();
        let load = src.vector_tile();
        comptime!(packing.physical(load.values()))
    }

    /// Scales a unit's fetched load lands its lines under ([`scales_per_load`](Tile::scales_per_load)).
    #[allow(dead_code)] // Reached through its expand, from `Stages::fetch_buffers`.
    pub(crate) fn decoded_scales_per_load(&self, src: &Tile) -> comptime_type!(usize) {
        let lands = self.vector_tile();
        src.scales_per_load(comptime!(lands.clone()))
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
        let per_load = self.decoded_scales_per_load(src);
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
                    let offset = comptime!(c * sw / per_load);
                    let at = line_coords(&start, &load, offset, &space);
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
        let per_load = self.decoded_scales_per_load(src);
        let space = comptime!(src.place.space.clone());
        let dst = comptime!(self.place.space.clone());
        let stored_as = src.packing();
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
        let through_products = reads_products::<O>(
            comptime!(LoadDecode {
                stored_as,
                loaded: sw,
                line: vw,
            }),
            mem,
        );
        let mut out = self.nd_mut::<O, VW>();
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let line = lines.region(t).coord(Level::COPIED) as u32;
                let start = load.start(line, &space);
                let held = raw[t];
                #[unroll]
                for s in 0..per_load {
                    let lines_under = comptime!(sw / vw / per_load);
                    let own = inner.scale_coords(
                        &line_coords(&start, &load, comptime!(s * lines_under * vw), &space),
                        &space,
                    );
                    let block = inner.scale_in_word::<f32>(scales[t * per_load + s], &own);
                    let decode = ScaledDecode::under::<O>(block * coarse, through_products);
                    #[unroll]
                    for c in 0..lines_under {
                        let offset = comptime!((s * lines_under + c) * vw);
                        let to = landing(&line_coords(&start, &load, offset, &space), &space, &dst);
                        out.write(
                            lands.index(&to, &dst),
                            decode.line::<S, O, I, WP, VW>(held, offset, mem, stored_as),
                        );
                    }
                }
            }
        }
    }
}

/// `at`, a value's coordinates over `space`, over the dimensions `dst` has.
#[cube]
fn landing(at: &Coords<u32>, #[comptime] space: &Space, #[comptime] dst: &Space) -> Coords<u32> {
    let mut to = Coords::<u32>::new();
    #[unroll]
    for p in 0..space.rank() {
        if dst.contains(space.dimension_at(p)) {
            to.push(at.at(p));
        }
    }
    to
}

/// The coordinates over `space` of the value `offset` into the load starting at `start`.
#[cube]
fn line_coords(
    start: &Coords<u32>,
    #[comptime] load: &VectorTile,
    #[comptime] offset: usize,
    #[comptime] space: &Space,
) -> Coords<u32> {
    let mut at = Coords::<u32>::new();
    #[unroll]
    for p in 0..space.rank() {
        let inner = comptime!(load.offset_along(offset, space.dimension_at(p)) as u32);
        at.push(start.at(p) + inner);
    }
    at
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
    #[comptime] offset: usize,
    mem: &Memory,
    #[comptime] stored_as: Packing,
    scale: f32,
) -> Vector<O, VW> {
    let packing = stored_as.placed();
    let lift = comptime!(packing.lift() as f32);
    let values = values_at::<S, I, WP, VW>(held, offset, packing);
    let scaled = Vector::<f32, VW>::cast_from(mem.codebook.entries(values))
        * Vector::<f32, VW>::cast_from(scale * lift);
    Vector::<O, VW>::cast_from(scaled)
}

/// `e2m1` codes a packed word holds.
const E2M1_CODES_PER_WORD: usize = 8;
/// Values of an 8-bit float a word holds.
const FP8_PER_WORD: usize = 4;
/// The magnitude of each `e2m1` code of a word, its sign, and the codes a byte apart they lie in.
const CODE_MAGNITUDES: u32 = 0x0707_0707;
const CODE_SIGNS: u32 = 0x0808_0808;
const CODE_BITS: u32 = 4;
/// Which of two words' bytes make the first and the second half of them interleaved.
const INTERLEAVED_LOW: u32 = 0x0501_0400;
const INTERLEAVED_HIGH: u32 = 0x0703_0602;

/// The products of the eight magnitudes of an `e2m1` code with `scale`, as `O`, 8 bits each: those
/// of codes 0 to 3 in the first word, 4 to 7 in the second.
#[cube(harness)]
fn e2m1_products<O: Numeric>(scale: f32) -> (u32, u32) {
    (
        e2m1_products_from::<O>(scale, 0usize),
        e2m1_products_from::<O>(scale, FP8_PER_WORD),
    )
}

/// [`e2m1_products`] of the four codes from `first` on.
#[cube]
fn e2m1_products_from<O: Numeric>(scale: f32, #[comptime] first: usize) -> u32 {
    let size!(W) = FP8_PER_WORD;
    let mut magnitudes = Vector::<f32, W>::empty();
    #[unroll]
    for code in 0..FP8_PER_WORD {
        let magnitude = comptime!(e2m1::from_bits((first + code) as u8).to_f32());
        magnitudes.insert(code, magnitude.runtime());
    }
    let products = Vector::<O, W>::cast_from(magnitudes * Vector::cast_from(scale));
    Vector::<u32, Const<1>>::reinterpret(products).extract(0usize)
}

/// The eight `e2m1` codes of `word`, the low nibble first, each as the byte that its magnitude's
/// product is, `products_low`'s for 0 to 3 and `products_high`'s for 4 to 7, under the code's sign:
/// codes 0 to 3 in the first word and 4 to 7 in the second.
#[cube(harness)]
fn e2m1_word_looked_up(word: u32, products_low: u32, products_high: u32) -> (u32, u32) {
    let (low, high) = (products_low, products_high);
    let odd = word >> CODE_BITS;
    // The even codes' bytes and the odd ones', each with its sign at the byte's top.
    let even_bytes =
        byte_permute(low, high, word & CODE_MAGNITUDES) | (word & CODE_SIGNS) << CODE_BITS;
    let odd_bytes =
        byte_permute(low, high, odd & CODE_MAGNITUDES) | (odd & CODE_SIGNS) << CODE_BITS;
    (
        byte_permute(even_bytes, odd_bytes, INTERLEAVED_LOW.runtime()),
        byte_permute(even_bytes, odd_bytes, INTERLEAVED_HIGH.runtime()),
    )
}

/// Whether `O` is an 8-bit float.
#[allow(clippy::extra_unused_type_parameters)] // Read by the expansion.
#[cube]
fn is_fp8<O: Numeric>() -> comptime_type!(bool) {
    intrinsic!(|scope| matches!(
        O::elem_type(scope),
        ElemType::Float(FloatKind::E4M3 | FloatKind::E5M2)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate cubecl as cubecl_runtime;
    use cubecl_runtime::TestRuntime;

    /// The sixteen `e2m1` codes of two words, each looked up in bytes that name its magnitude.
    #[test]
    fn every_e2m1_code_is_its_magnitudes_byte_under_its_sign() {
        for (w, word) in [0x76543210u32, 0xFEDCBA98u32].into_iter().enumerate() {
            let (low, high) =
                e2m1_word_looked_up::harness::<TestRuntime>(word, 0x13121110, 0x17161514);
            let got: Vec<u8> = [low, high].iter().flat_map(|b| b.to_le_bytes()).collect();
            let sign = (w as u8) << 7;
            let want: Vec<u8> = (0..8).map(|code| (0x10 + code) | sign).collect();
            assert_eq!(got, want, "word {word:#010x}");
        }
    }
}
