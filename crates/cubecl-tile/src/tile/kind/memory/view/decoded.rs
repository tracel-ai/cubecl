//! The decoding read: a tile's window served into registers a line at a time, each line decoded
//! as [`dequantize_from`](Tile::dequantize_from) writes it ([`Tile::decoded`]).

use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::layout::Coords1d;

use crate::*;

/// The lines of a tile's window at `O`, `W` values a line, each read decoded ([`Tile::decoded`]).
///
/// A line is named by its row, the window's lines grouped by every axis but the innermost and
/// counted row-major, and its place in the row. Every line of a row shares the row's address and
/// scale, so a reader taking several lines of one row computes them once.
pub struct Decoded<'a, O: Float, W: Size> {
    _lines: PhantomData<(&'a (), O, W)>,
}

/// How a [`Decoded`] read is served, settled as the kernel expands.
pub enum DecodedExpand<'a, O: Float, W: Size> {
    /// Plain values under no scale: read as they lie, as [`flat`](Tile::flat) reads them, a row
    /// `lines` long.
    Plain {
        values: MaskedExpand<'a, Vector<O, W>, Coords1d>,
        lines: usize,
    },
    /// Packed values: the window, its words unpacked, looked up and scaled at each read.
    Packed(Box<TileExpand>),
}

impl<O: Float, W: Size> Decoded<'_, O, W> {
    /// The `line`-th line of row `row`, masked where it overhangs the operand.
    pub fn read(&self, _row: usize, _line: usize) -> Vector<O, W> {
        unexpanded!()
    }

    /// The `line`-th line of row `row`, untested: the reader proved it in bounds, so reading past
    /// them is out of bounds, not masked. What a reader that tests each line itself takes, so the
    /// read does not test it again.
    ///
    /// # Panics
    ///
    /// As the kernel expands, on a window with a [`Boundary::Clamp`](crate::kind::Boundary::Clamp)
    /// axis, whose reads must be clamped.
    pub fn read_unchecked(&self, _row: usize, _line: usize) -> Vector<O, W> {
        unexpanded!()
    }
}

impl<O: Float, W: Size> DecodedExpand<'_, O, W> {
    pub fn __expand_read_method(
        &self,
        scope: &Scope,
        row: NativeExpand<usize>,
        line: NativeExpand<usize>,
    ) -> NativeExpand<Vector<O, W>> {
        self.read(scope, row, line, Guard::Checked)
    }

    pub fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        row: NativeExpand<usize>,
        line: NativeExpand<usize>,
    ) -> NativeExpand<Vector<O, W>> {
        if let DecodedExpand::Packed(window) = self {
            assert!(
                window.__expand_guard_provable_method(scope),
                "Decoded::read_unchecked: a clamped window's reads are clamped, never untested"
            );
        }
        self.read(scope, row, line, Guard::Proved)
    }

    fn read(
        &self,
        scope: &Scope,
        row: NativeExpand<usize>,
        line: NativeExpand<usize>,
        guard: Guard,
    ) -> NativeExpand<Vector<O, W>> {
        match self {
            DecodedExpand::Plain { values, lines } => {
                read_plain_line::expand::<O, W>(scope, values, row, line, *lines, guard)
            }
            DecodedExpand::Packed(window) => {
                read_packed_line::expand::<O, W>(scope, window, row, line, guard)
            }
        }
    }
}

impl<'a, O: Float, W: Size> CubeType for Decoded<'a, O, W> {
    type ExpandType = DecodedExpand<'a, O, W>;
}

impl<O: Float, W: Size> IntoExpand for DecodedExpand<'_, O, W> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}

impl<O: Float, W: Size> ExpandTypeClone for DecodedExpand<'_, O, W> {
    fn clone_unchecked(&self) -> Self {
        match self {
            DecodedExpand::Plain { values, lines } => DecodedExpand::Plain {
                values: values.clone_unchecked(),
                lines: *lines,
            },
            DecodedExpand::Packed(window) => DecodedExpand::Packed(window.clone()),
        }
    }
}

impl<O: Float, W: Size> IntoMut for DecodedExpand<'_, O, W> {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl<O: Float, W: Size> CubeDebug for DecodedExpand<'_, O, W> {}

impl<O: Float, W: Size> AsRefExpand for DecodedExpand<'_, O, W> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl<O: Float, W: Size> AsMutExpand for DecodedExpand<'_, O, W> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl Tile {
    /// This tile's window read into registers a line at a time, `W` values a line, each served
    /// at `O` decoded as [`dequantize_from`](Tile::dequantize_from) writes it: its fields
    /// unpacked, looked up in the table they index, under the scales they carry. What a unit
    /// reading the lines it needs straight out of memory takes, with no stage to decode into.
    ///
    /// Plain values under no scale read as [`flat`](Tile::flat) reads them. Packed values are read
    /// a word a load, so they are bound one word wide; a line takes the words it spans, or the
    /// fields it needs of one, and their scale, which must not vary along the line.
    ///
    /// # Panics
    ///
    /// As the kernel expands: plain values under scales, which only a copy decodes; packed ones
    /// bound wider than a word, or whose scales vary along a line.
    pub fn decoded<O: Float, W: Size>(&self) -> Decoded<'_, O, W> {
        unexpanded!()
    }
}

impl TileExpand {
    pub fn __expand_decoded_method<O: Float, W: Size>(
        &self,
        scope: &Scope,
    ) -> DecodedExpand<'_, O, W> {
        match self.__expand_packing_method(scope) {
            Packing::Plain => {
                assert!(
                    !self.factor.scaled(),
                    "Tile::decoded: plain values under scales are decoded by a copy \
                     (`dequantize_from`)"
                );
                let space = &self.place.space;
                let row = space.extent_at(space.rank() - 1);
                DecodedExpand::Plain {
                    values: self.__expand_flat_method::<O, W>(scope),
                    lines: row / W::__expand_value(scope),
                }
            }
            Packing::Packed { .. } => DecodedExpand::Packed(Box::new(self.clone())),
        }
    }
}

/// The `line`-th line of row `row` of plain `values`, `lines` a row, read under `guard`.
#[cube]
fn read_plain_line<O: Float, W: Size>(
    values: &Masked<'_, Vector<O, W>, Coords1d>,
    row: usize,
    line: usize,
    #[comptime] lines: usize,
    #[comptime] guard: Guard,
) -> Vector<O, W> {
    match comptime!(guard) {
        Guard::Checked => values.read(row * lines + line),
        Guard::Proved => values.read_unchecked(row * lines + line),
    }
}

/// The `line`-th line of row `row` of a packed `window`, `W` values from the row's start: the
/// words it spans, or the fields it needs of one, each field looked up in the table it indexes
/// where it has one, and the line multiplied by its scale in `f32`, rounded to `O` once. Its
/// words are read under `guard`.
#[cube]
fn read_packed_line<O: Float, W: Size>(
    window: &Tile,
    row: usize,
    line: usize,
    #[comptime] guard: Guard,
) -> Vector<O, W> {
    let space = comptime!(window.place.space.clone());
    let rank = comptime!(space.rank());
    let along = comptime!(space.axis_at(rank - 1));
    let packing = window.packing();
    let field = comptime!(match packing {
        Packing::Packed { field } => field,
        Packing::Plain => unreachable!("Tile::decoded serves plain values as they lie"),
    });
    let (bits, per_word) = comptime!((field.size_bits() as u32, field.per_word()));
    let width = W::value();
    let load = window.vector_tile();
    comptime!(assert!(
        load.extents() == [(along, per_word)],
        "Tile::decoded: packed values are read a word a load, along {along:?}; this tile loads \
         {:?}. Bind it one word wide",
        load.extents()
    ));
    let varies = window.factor.varies_along(comptime!(along));
    comptime!(assert!(
        width == 1 || !varies,
        "Tile::decoded: a {width}-wide line runs along {along:?}, which the scales vary along, so \
         one line lies under several scales"
    ));
    // A line holds whole words, or sits inside one: both are powers of two.
    let fields = comptime!(core::cmp::min(width, per_word));
    let words_per_line = comptime!(width / fields);
    let size!(NF) = fields;
    let mem = window.mem("Tile::decoded");
    let looks_up = mem.codebook.present();
    let scaled = window.factor.scaled();
    let words = window.nd_words::<Const<1>>(guard);
    // The row's coordinates, every axis but the innermost; then where the line starts along it.
    let rows = Coords::constant(comptime!(
        (0..rank - 1)
            .map(|p| space.extent_at(p))
            .collect::<Vec<_>>()
    ));
    let mut first = rows.unravel(row as u32);
    let along_first = (line * width) as u32;
    first.push(along_first);
    // Where the line starts in its word: zero where it spans whole words.
    let shift = (along_first % comptime!(per_word as u32)) * bits;

    let mut values = Vector::<f32, W>::empty();
    #[unroll]
    for w in 0..words_per_line {
        let mut at = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            if comptime!(p == rank - 1) {
                at.push(first.at(p) + comptime!((w * per_word) as u32));
            } else {
                at.push(first.at(p));
            }
        }
        let mut word = words.read(load.index(&at, &space)).extract(0usize);
        if comptime!(fields < per_word) {
            word >>= shift;
        }
        let word = Vector::<u32, Const<1>>::new(word);
        if comptime!(looks_up) {
            let indices = unpack_line::<u32, Const<1>, NF>(word, comptime!(field));
            #[unroll]
            for j in 0..fields {
                values.insert(w * fields + j, mem.codebook.entry(indices.extract(j)));
            }
        } else {
            let fields_of_word = unpack_line::<f32, Const<1>, NF>(word, comptime!(field));
            #[unroll]
            for j in 0..fields {
                values.insert(w * fields + j, fields_of_word.extract(j));
            }
        }
    }
    if comptime!(scaled) {
        let scale = window
            .factor
            .at_coords_under(&first, comptime!(space.clone()), guard);
        values *= Vector::<f32, W>::cast_from(scale);
    }
    Vector::<O, W>::cast_from(values)
}
