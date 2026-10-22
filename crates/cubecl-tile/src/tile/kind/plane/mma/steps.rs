//! A manual-mma fragment deeper along `k` than its instruction, loaded out of shared memory one
//! instruction step at a time, each step's registers in the instruction's own order.
//!
//! [Lane runs](super::lane_runs) reorder `k` so a unit reads its share in whole loads, which pays
//! in global memory. Out of shared memory a plain step loads as an instruction-deep fragment does,
//! through `ldmatrix` where the device offers it.
//!
//! A weight of `e2m1` codes under words of four `e4m3` block scales (NVFP4 as the block-scaled
//! instruction reads it) decodes as it loads instead. Under the 16-bit `m16n8k16` layout a unit of
//! quad place `q` holds, in register `i` of step `s`, the pair of codes at `2q + 8i + 16s` of its
//! column: byte `q` of word `2s + i` of the column's codes, under field `s % 4` of the column's
//! scale word for that step's 64 of `k`.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2, e4m3,
    prelude::*,
    quant::scheme::QuantValue,
};

use super::block_scaled::{E2M1_PER_WORD, E2M1_SCALE_BLOCK};
use super::lane_runs::{instruction_depth, lays_out_lane_runs};
use crate::*;

/// `e4m3` block scales one word of them holds.
const SCALES_PER_WORD: usize = 4;

/// Why `src`, a shared-memory window of an `m × n × k` fragment of `T` in role `ident`, cannot
/// load a step at a time; `None` where it can: a fragment deeper than its instruction over a plain
/// window, each step read as a fragment of its own at its offset along `k`, whatever layout the
/// target's registers take; or deeper than a 16-bit `m16n8k16` instruction, on a target that lays
/// its registers out as [`lays_out_lane_runs`] reads them, over a `B` of `e2m1` codes along `k`
/// under one level of words of block scales, a whole number of words deep so every window starts
/// on a word.
#[cube]
pub(crate) fn steps_refusal<T: Numeric>(
    src: &Tile,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
) -> comptime_type!(Option<&'static str>) {
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let laid_out = lays_out_lane_runs();
    let bytes = T::size().comptime();
    let packing = src.packing();
    let levels = src.factor_levels();
    let words = src.innermost_factor().holds_words();
    let e2m1_weight = comptime!(
        ident == MatrixIdent::B
            && layout == MatrixLayout::ColMajor
            && matches!(
                packing,
                Packing::Packed {
                    field: Field::Quant(QuantValue::E2M1) | Field::ConvertedE2M1
                }
            )
    );
    comptime!(if k <= depth {
        Some("the fragment is one instruction deep, which loads as one")
    } else if packing == Packing::Plain && levels == 0 {
        None
    } else if (m, n, depth) != (16, 8, 16) || bytes != 2 || !laid_out {
        Some("a step's registers are read off the 16-bit m16n8k16 layout, which this is not")
    } else if !e2m1_weight || levels != 1 || !words {
        Some(
            "a packed window loads a step at a time only as a B of e2m1 codes along `k` under \
             words of e4m3 block scales",
        )
    } else if !k.is_multiple_of(E2M1_SCALE_BLOCK * SCALES_PER_WORD) {
        Some("a window under words of block scales is a whole number of words deep")
    } else {
        None
    })
}

/// Fill `fragment`, `ident`'s registers of every step one after another, from `src`, a
/// shared-memory window `shape` deep.
#[cube]
pub(crate) fn load_in_steps<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] shape: (usize, usize, usize),
) {
    let refusal = steps_refusal::<T>(src, ident, layout, shape);
    comptime!(if let Some(why) = refusal {
        panic!(
            "MmaData::load: a fragment deeper than its instruction loads a step at a time: {why}"
        )
    });
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let def = MmaDefinition::<T, T, T>::new(m, n, depth);
    let per_step = def.vectors_per_lane(ident);
    let edges = comptime!(match ident {
        MatrixIdent::A => (m, k),
        _ => (k, n),
    });
    let packing = src.packing();
    match comptime!(packing) {
        Packing::Plain =>
        {
            #[unroll]
            for s in 0..comptime!(k / depth) {
                let mut step = Array::<Vector<T, N>>::new(per_step);
                let offset = comptime!(s * depth);
                load_fragment(src, &mut step, &def, ident, layout, io, edges, offset);
                #[unroll]
                for i in 0..per_step {
                    fragment[comptime!(s * per_step + i)] = step[i];
                }
            }
        }
        Packing::Packed { .. } => load_e2m1_weight::<T, N>(src, fragment, &def, k, n),
    }
}

/// Every step of a `B` of `e2m1` codes under words of block scales, decoded into the
/// instruction's registers: the unit reads its column's codes and scale words for the whole
/// window once, then decodes one byte of codes a register.
#[cube]
fn load_e2m1_weight<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<T, T, T>,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    let space = comptime!(src.place.space.clone());
    // A col-major `B`'s rows are its columns.
    let axes = comptime!(MatrixAxes::new(&space, n, k).unwrap_or_else(|no| {
        panic!("MmaData::load: a stepped weight is an {n} x {k} matrix: {no:?}")
    }));
    let load = src.vector_tile();
    let values = comptime!(load.values());
    comptime!(assert!(
        values.is_multiple_of(E2M1_PER_WORD) && k.is_multiple_of(values),
        "MmaData::load: a stepped weight is read in whole words, whole lines of its {k} of `k`, \
         and this one {values} values a load"
    ));
    let per_line = comptime!(values / E2M1_PER_WORD);
    let size!(WP) = per_line;
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let factor = src.innermost_factor();

    // The unit's column, and its byte of every word as a shift: a runtime index into a vector of
    // registers lowers to local memory, a shift stays in registers.
    let (first, column) = def.position_of_nth(UNIT_POS_PLANE, 0u32, MatrixIdent::B);
    let shift = first / 2u32 * 8u32;

    let mut codes = Array::<Vector<u32, WP>>::new(comptime!(k / values));
    #[unroll]
    for l in 0..comptime!(k / values) {
        let at = TileMatrix::value_coords(
            column,
            comptime!((l * values) as u32),
            0usize,
            &space,
            axes,
            1usize,
        );
        codes[l] = words.read(load.index(&at, &space));
    }
    let scale_words = comptime!(k / (E2M1_SCALE_BLOCK * SCALES_PER_WORD));
    // Each word's four scales converted at once, the device's pair conversion taking two a
    // call, rather than a field selected and converted for every step it serves.
    let mut scales = Array::<Vector<T, Const<4>>>::new(scale_words);
    #[unroll]
    for w in 0..scale_words {
        let at = TileMatrix::value_coords(
            column,
            comptime!((w * E2M1_SCALE_BLOCK * SCALES_PER_WORD) as u32),
            0usize,
            &space,
            axes,
            1usize,
        );
        let word = factor.word_at(&at, comptime!(space.clone()));
        scales[w] = Vector::<T, Const<4>>::cast_from(Vector::<e4m3, Const<4>>::reinterpret(word));
    }

    #[unroll]
    for s in 0..comptime!(k / E2M1_SCALE_BLOCK) {
        let scale = scales[comptime!(s / SCALES_PER_WORD)].extract(comptime!(s % SCALES_PER_WORD));
        #[unroll]
        for i in 0..2usize {
            let word = comptime!(2 * s + i);
            let line = codes[comptime!(word / per_line)].extract(comptime!(word % per_line));
            let pair = Vector::<e2m1x2, Const<4>>::reinterpret(line >> shift).extract(0usize);
            let decoded = Vector::<T, Const<2>>::cast_from(Vector::<e2m1x2, Const<1>>::new(pair))
                * Vector::<T, Const<2>>::new(scale);
            let mut cells = Vector::<T, N>::empty();
            cells.insert(0usize, decoded.extract(0usize));
            cells.insert(1usize, decoded.extract(1usize));
            fragment[comptime!(2 * s + i)] = cells;
        }
    }
}
