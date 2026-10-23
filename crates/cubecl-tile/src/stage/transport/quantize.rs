//! The encoding copy: a float tile written packed, under the block scales it computes
//! ([`Tile::quantize_from`]).

use cubecl::ir::ElemType;
use cubecl::ir::types::Fp8Format;
use cubecl::post_processing::minifloat::{f32_to_fp8_bits, f32_to_ue8m0_bits};
use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantValue, ScaleDtype};
use cubecl_std::quant::fp4::float_to_e2m1_bits;
use cubecl_std::quant::round::round_up_to_dtype;

use crate::*;

#[cube]
impl Tile {
    /// [`quantize_from`](Tile::quantize_from) a float source in memory: each load of the
    /// innermost scales is one unit's, which writes the blocks those scales cover. A block is every
    /// value at a scale's coordinates along the axes the scales span, and every position along the
    /// axes they omit. It is read in the source's loads and written in this tile's, each a whole
    /// number of the source's.
    pub(crate) fn copy_quantized_from<F: Float>(&mut self, src: &Tile) {
        let mut scales = self.factor.innermost_scale();
        let coarse = self.factor.coarse();
        let two_levels = comptime!(self.factor.depth() > 1);
        let values_space = comptime!(self.place.space.clone());
        let scales_space = comptime!(scales.place.space.clone());
        let block_space = comptime!(values_space.subspace(&values_space.difference(&scales_space)));
        let load = self.vector_tile();
        let src_load = src.vector_tile();
        let scale_load = scales.vector_tile();
        comptime!(check_encoding_copy(
            &values_space,
            &block_space,
            &load,
            &src_load
        ));
        let size!(SW) = comptime!(src_load.values());
        let packing = self.packing();
        let size!(WP) = comptime!(packing.physical(load.values()));
        let value = comptime!(match packing {
            Packing::Packed { field } => field.quantized_value(),
            Packing::Plain => panic!("Tile::quantize_from: the values are written packed"),
        });
        let scale_packing = scales.packing();
        let dtype = comptime!(match scale_packing {
            Packing::Packed { field } => field.scale_dtype(),
            Packing::Plain => match scales.element {
                ElemType::Float(kind) => Field::of_float(kind).scale_dtype(),
                other => panic!("Tile::quantize_from: a scale is a float, not {other:?}"),
            },
        });
        let (low, high) = comptime!(value.range());
        let largest = comptime!(((high - low) / 2.0) as f64);
        let src_lines = comptime!(block_space.tiles(&src_load.space()));
        let loads_a_block = comptime!(block_space.tiles(&load.space()));
        let lines_a_load = comptime!(load.values() / src_load.values());
        let scales_a_load = comptime!(scale_load.values());
        let scale_loads = comptime!(scale_load.counts(&scales_space).iter().product::<usize>());

        let fill = self.fill_units();
        let read = src.nd_packed::<F, SW>(comptime!(Guard::Checked));
        let mut words = self.nd_words_mut::<WP>();
        let first = FillUnits::worker(fill) as u32;
        let stride = FillUnits::workers(fill) as u32;
        for line in range_stepped(first, comptime!(scale_loads as u32), stride) {
            let scales_start = scale_load.start(line, &scales_space);
            let mut scale_codes = Vector::<u32, Const<1>>::new(0u32);
            #[unroll]
            for c in 0..scales_a_load {
                // The block's first value: the scale's coordinates, zero along the omitted axes.
                let mut origin = Coords::<u32>::new();
                #[unroll]
                for p in 0..comptime!(values_space.rank()) {
                    let axis = comptime!(values_space.axis_at(p));
                    if comptime!(scales_space.contains(axis)) {
                        let at = comptime!(scales_space.position(axis));
                        let inside = comptime!(scale_load.offset_along(c, axis) as u32);
                        origin.push(scales_start.at(at) + inside);
                    } else {
                        origin.push(0u32);
                    }
                }
                let mut lines = Array::<Vector<F, SW>>::new(src_lines);
                let mut most = F::new(0.0);
                #[unroll]
                for l in 0..src_lines {
                    let at = Self::block_load(
                        &origin,
                        comptime!(l as u32),
                        comptime!(values_space.clone()),
                        comptime!(block_space.clone()),
                        comptime!(src_load.clone()),
                    );
                    let held = read.read(src_load.index(&at, &values_space));
                    lines[l] = held;
                    #[unroll]
                    for e in 0..comptime!(src_load.values()) {
                        most = max(most, F::abs(held.extract(e)));
                    }
                }
                // As burn calibrates it: the block's largest over the format's, in the source's
                // element, rounded up to the scale's dtype, which never clips the block's largest.
                // Under a coarser scale the block's is taken relative to it, in `f32`.
                let wanted = most / F::new(largest);
                let inner = if comptime!(two_levels) {
                    round_up_to_dtype::<F>(F::cast_from(f32::cast_from(wanted) / coarse), dtype)
                } else {
                    round_up_to_dtype::<F>(wanted, dtype)
                };
                // An all-zero block has a zero scale, which no value is over: its codes are zero.
                let empty = most == F::new(0.0);
                let mut codes = Array::<Vector<u32, SW>>::new(src_lines);
                #[unroll]
                for l in 0..src_lines {
                    let scaled = if comptime!(two_levels) {
                        Vector::<f32, SW>::cast_from(lines[l])
                            / Vector::new(coarse * f32::cast_from(inner))
                    } else {
                        Vector::<f32, SW>::cast_from(lines[l] / Vector::new(inner))
                    };
                    let encoded = Self::encode_values::<SW>(scaled, value);
                    codes[l] = select(empty, Vector::new(0u32), encoded);
                }
                #[unroll]
                for j in 0..loads_a_block {
                    let at = Self::block_load(
                        &origin,
                        comptime!(j as u32),
                        comptime!(values_space.clone()),
                        comptime!(block_space.clone()),
                        comptime!(load.clone()),
                    );
                    words.write(
                        load.index(&at, &values_space),
                        Self::pack_codes::<SW, WP>(
                            &codes,
                            comptime!(j * lines_a_load),
                            comptime!(value.size_bits()),
                            comptime!(src_load.values()),
                            comptime!(packing.physical(load.values())),
                        ),
                    );
                }
                Self::write_scale::<F>(
                    &mut scales,
                    &mut scale_codes,
                    inner,
                    &scales_start,
                    comptime!(c),
                    scale_packing,
                );
            }
            if comptime!(matches!(scale_packing, Packing::Packed { .. })) {
                let mut scale_words = scales.nd_words_mut::<Const<1>>();
                scale_words.write(scale_load.index(&scales_start, &scales_space), scale_codes);
            }
        }
    }

    /// The first value of the `j`-th of a block's loads, the block starting at `origin`: the
    /// loads walk the block's space as they walk any window's, the last axis fastest.
    fn block_load(
        origin: &Coords<u32>,
        #[comptime] j: u32,
        #[comptime] values: Space,
        #[comptime] block: Space,
        #[comptime] load: VectorTile,
    ) -> Coords<u32> {
        let inside = load.start(comptime!(j).runtime(), &block);
        let mut at = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(values.rank()) {
            let axis = comptime!(values.axis_at(p));
            if comptime!(block.contains(axis)) {
                at.push(origin.at(p) + inside.at(comptime!(block.position(axis))));
            } else {
                at.push(origin.at(p));
            }
        }
        at
    }

    /// Each of `values`, already over its scale, as the code `value` stores it in, in the low
    /// bits: a float format rounded to its nearest code, ties to even, saturating; an integer one
    /// rounded to the nearest integer and clamped to its range, in two's complement.
    fn encode_values<N: Size>(
        values: Vector<f32, N>,
        #[comptime] value: QuantValue,
    ) -> Vector<u32, N> {
        let (low, high) = comptime!(value.range());
        let clamped = clamp(values, Vector::new(low), Vector::new(high));
        #[comptime]
        match value {
            QuantValue::E2M1 => float_to_e2m1_bits::<f32, N>(clamped),
            QuantValue::E4M3 => f32_to_fp8_bits::<N>(clamped, Fp8Format::E4M3),
            QuantValue::E5M2 => f32_to_fp8_bits::<N>(clamped, Fp8Format::E5M2),
            _ => {
                let rounded = clamp(Vector::round(values), Vector::new(low), Vector::new(high));
                Vector::<u32, N>::reinterpret(Vector::<i32, N>::cast_from(rounded))
            }
        }
    }

    /// The `wp` words of one of this tile's loads, packed from `codes`, the block's codes `bits`
    /// wide, `sw` a source line, starting at its `first` line: each word's first code in its low
    /// bits.
    fn pack_codes<SW: Size, WP: Size>(
        codes: &Array<Vector<u32, SW>>,
        #[comptime] first: usize,
        #[comptime] bits: usize,
        #[comptime] sw: usize,
        #[comptime] wp: usize,
    ) -> Vector<u32, WP> {
        let per_word = comptime!(32 / bits);
        let mask = comptime!(((1u64 << bits) - 1) as u32);
        let mut words = Vector::<u32, WP>::empty();
        #[unroll]
        for w in 0..wp {
            let mut word = 0u32;
            #[unroll]
            for i in 0..per_word {
                let at = comptime!(first * sw + w * per_word + i);
                let code = codes[comptime!(at / sw)].extract(comptime!(at % sw));
                word |= (code & mask) << comptime!((i * bits) as u32);
            }
            words.insert(w, word);
        }
        words
    }

    /// Writes `scale`, the `c`-th of the scale load starting at `start`: into `codes`, the load's
    /// one word, where the scales are packed and the caller writes the word whole, or straight into
    /// `scales` where they are stored as plain floats.
    fn write_scale<F: Float>(
        scales: &mut Tile,
        codes: &mut Vector<u32, Const<1>>,
        scale: F,
        start: &Coords<u32>,
        #[comptime] c: usize,
        #[comptime] packing: Packing,
    ) {
        #[comptime]
        match packing {
            Packing::Packed { field } => {
                // Exact: the scale is already rounded onto its dtype's grid.
                let rounded = Vector::<f32, Const<1>>::new(f32::cast_from(scale));
                let code = match comptime!(field.scale_dtype()) {
                    ScaleDtype::UE8M0 => f32_to_ue8m0_bits::<Const<1>>(rounded),
                    _ => f32_to_fp8_bits::<Const<1>>(rounded, Fp8Format::E4M3),
                };
                let shift = comptime!((c * field.size_bits()) as u32);
                codes.insert(
                    0usize,
                    codes.extract(0usize) | (code.extract(0usize) << shift),
                );
            }
            Packing::Plain => {
                let space = comptime!(scales.place.space.clone());
                let load = scales.vector_tile();
                let define!(Stored) = scales.element;
                let mut out = scales.nd_mut::<Stored, Const<1>>();
                let mut at = Coords::<u32>::new();
                #[unroll]
                for p in 0..comptime!(space.rank()) {
                    let inside = comptime!(load.offset_along(c, space.axis_at(p)) as u32);
                    at.push(start.at(p) + inside);
                }
                out.write(
                    load.index(&at, &space),
                    Vector::new(Stored::cast_from(scale)),
                );
            }
        }
    }
}

/// What an encoding copy into `values`, written in `load`s from a source read in `src` loads,
/// rests on. A block is `block`, the values' axes its scales omit.
fn check_encoding_copy(values: &Space, block: &Space, load: &VectorTile, src: &VectorTile) {
    assert!(
        values.axes().all(|axis| !values.is_dynamic(axis)),
        "Tile::quantize_from: an encoding copy walks the values' blocks, which needs their extents \
         at expansion; this window spans a dynamic axis"
    );
    for tile in [load, src] {
        assert!(
            tile.extents().iter().all(|&(axis, _)| block.contains(axis))
                && block.tiles(&tile.space()) * tile.values() == block.cells(),
            "Tile::quantize_from: a block of {block:?} is whole loads of {:?}, each inside it",
            tile.extents()
        );
    }
    let (along, run) = load.along_one_axis("Tile::quantize_from");
    let (src_along, src_run) = src.along_one_axis("Tile::quantize_from");
    assert!(
        along == src_along && run.is_multiple_of(src_run),
        "Tile::quantize_from: a written load is whole source loads along one axis, and {run} \
         along {along:?} is not of {src_run} along {src_along:?}"
    );
}
