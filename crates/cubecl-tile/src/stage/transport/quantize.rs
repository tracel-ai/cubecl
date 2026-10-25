//! The encoding copy: a float tile written packed, under the block scales it computes
//! ([`Tile::quantize_from`]).

use cubecl::ir::ElemType;
use cubecl::ir::types::Fp8Format;
use cubecl::post_processing::minifloat::{f32_to_fp8_bits, f32_to_ue8m0_bits};
use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantValue, ScaleDtype};
use cubecl_std::quant::fp4::float_to_e2m1_bits;
use cubecl_std::quant::round::round_up_to_dtype;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// What an encoding copy reads off its tiles at expansion: the spaces it walks, the loads it reads
/// and writes in, and the formats it writes ([`Tile::copy_quantized_from`]).
#[derive(Clone, Debug)]
pub(crate) struct EncodingCopy {
    /// The values' window.
    values: Space,
    /// The source's window: the values' axes, but for ones of extent one.
    source: Space,
    /// The innermost scales' window.
    scales: Space,
    /// One block: the values' axes the scales omit.
    block: Space,
    /// A written load of the values, a read of the source, a load of the scales.
    load: VectorTile,
    line: VectorTile,
    scale_load: VectorTile,
    /// The values' packing and the number format each value is written in.
    packing: Packing,
    value: QuantValue,
    /// How the scales are stored, and the dtype each is rounded to.
    scale_packing: Packing,
    dtype: ScaleDtype,
    /// Whether a coarser scale sits over the block scales.
    two_levels: bool,
}

impl EncodingCopy {
    /// Lines of the source one block holds.
    fn lines_a_block(&self) -> usize {
        self.block.tiles(&self.line.space())
    }

    /// Lines of the source one written load holds.
    fn lines_a_load(&self) -> usize {
        self.load.values() / self.line.values()
    }

    /// Lines of the source one load of scales covers: every line of every block it scales.
    fn lines_a_scale_load(&self) -> usize {
        self.scale_load.values() * self.lines_a_block()
    }

    /// Loads of scales the window holds.
    fn scale_loads(&self) -> usize {
        self.scale_load.counts(&self.scales).iter().product()
    }

    /// Whether a plane's lanes can share the copy, a line a lane: a block's, a written load's and
    /// a load of scales' lines are each a power of two, so each group of lanes that meets through
    /// the plane is aligned in it.
    fn lanes_can_share(&self) -> bool {
        [
            self.lines_a_block(),
            self.lines_a_load(),
            self.lines_a_scale_load(),
        ]
        .iter()
        .all(|lines| lines.is_power_of_two())
    }
    /// What the copy rests on: the values' extents known at expansion, the source over the
    /// values' axes but for ones of extent one, every load and line inside one block, the written
    /// loads whole lines along the one axis both run along, and the scales loaded along one axis.
    fn check(&self) {
        let (values, source, block) = (&self.values, &self.source, &self.block);
        assert!(
            values.axes().all(|axis| !values.is_dynamic(axis)),
            "Tile::quantize_from: an encoding copy walks the values' blocks, which needs their \
             extents at expansion; this window spans a dynamic axis"
        );
        assert!(
            source.axes().all(|axis| values.contains(axis))
                && values
                    .axes()
                    .all(|axis| source.contains(axis) || values.extent(axis) == 1),
            "Tile::quantize_from: the source spans the values' axes, but for one of extent one: \
             {source:?} against {values:?}"
        );
        for tile in [&self.load, &self.line] {
            assert!(
                tile.extents().iter().all(|&(axis, _)| block.contains(axis))
                    && block.tiles(&tile.space()) * tile.values() == block.cells(),
                "Tile::quantize_from: a block of {block:?} is whole loads of {:?}, each inside it",
                tile.extents()
            );
        }
        // A block's offset in a load of scales is its place along the one axis the load runs on.
        self.scale_load.along_one_axis("Tile::quantize_from");
        let (along, run) = self.load.along_one_axis("Tile::quantize_from");
        let (line_along, line_run) = self.line.along_one_axis("Tile::quantize_from");
        assert!(
            along == line_along && run.is_multiple_of(line_run),
            "Tile::quantize_from: a written load is whole source lines along one axis, and {run} \
             along {along:?} is not of {line_run} along {line_along:?}"
        );
    }
}

#[cube]
impl Tile {
    /// [`quantize_from`](Tile::quantize_from) a float source in memory. A block is every value at
    /// a scale's coordinates along the axes the scales span, and every position along the axes
    /// they omit; it is read in the source's lines and written in this tile's loads, each a whole
    /// number of lines.
    ///
    /// A window a plane holds is quantized by its lanes together, a line a lane, where the plane
    /// is as wide as a load of scales' lines ([`quantize_across_lanes`](Tile::quantize_across_lanes));
    /// any other, a load of scales a unit ([`quantize_per_unit`](Tile::quantize_per_unit)).
    pub(crate) fn copy_quantized_from<F: Float>(&mut self, src: &Tile) {
        let mut scales = self.factor.innermost_scale();
        let coarse = self.factor.coarse();
        let values = comptime!(self.place.space.clone());
        let scales_space = comptime!(scales.place.space.clone());
        let packing = self.packing();
        let scale_packing = scales.packing();
        let (load, line, scale_load) =
            (self.vector_tile(), src.vector_tile(), scales.vector_tile());
        let plan = comptime!(EncodingCopy {
            values: values.clone(),
            source: src.place.space.clone(),
            scales: scales_space.clone(),
            block: values.subspace(&values.difference(&scales_space)),
            load,
            line,
            scale_load,
            packing,
            value: match packing {
                Packing::Packed { field } => field.quantized_value(),
                Packing::Plain => panic!("Tile::quantize_from: the values are written packed"),
            },
            scale_packing,
            dtype: match scale_packing {
                Packing::Packed { field } => field.scale_dtype(),
                Packing::Plain => match scales.element {
                    ElemType::Float(kind) => Field::of_float(kind).scale_dtype(),
                    other => panic!("Tile::quantize_from: a scale is a float, not {other:?}"),
                },
            },
            two_levels: self.factor.depth() > 1,
        });
        comptime!(plan.check());
        // A window its holder shares is the holder's units' to fill, as a plain copy's is.
        let holder = comptime!(self.place.holder());
        let fill = self.fill_units();
        let fill = comptime!(fill.held_by(holder));
        // A plane narrower than a load of scales' lines cannot meet them through itself.
        let lanes_share = comptime!(fill.scope == ComputeScope::Plane && plan.lanes_can_share());
        if lanes_share
            && FillUnits::workers(fill) as u32 >= comptime!(plan.lines_a_scale_load() as u32)
        {
            self.quantize_across_lanes::<F>(src, &mut scales, coarse, fill, plan.clone());
        } else {
            self.quantize_per_unit::<F>(src, &mut scales, coarse, fill, plan);
        }
    }

    /// Each load of the innermost scales a unit's, which writes the blocks those scales cover.
    fn quantize_per_unit<F: Float>(
        &mut self,
        src: &Tile,
        scales: &mut Tile,
        coarse: f32,
        #[comptime] fill: FillUnits,
        #[comptime] plan: EncodingCopy,
    ) {
        let size!(SW) = comptime!(plan.line.values());
        let size!(WP) = comptime!(plan.packing.physical(plan.load.values()));
        let src_lines = comptime!(plan.lines_a_block());
        let loads_a_block = comptime!(plan.block.tiles(&plan.load.space()));
        let scales_a_load = comptime!(plan.scale_load.values());
        let scale_loads = comptime!(plan.scale_loads());
        let read = src.nd_packed::<F, SW>(comptime!(Guard::Checked));
        let mut words = self.nd_words_mut::<WP>();
        let first = FillUnits::worker(fill) as u32;
        let stride = FillUnits::workers(fill) as u32;
        for at_load in range_stepped(first, comptime!(scale_loads as u32), stride) {
            let scales_start = comptime!(plan.scale_load.clone()).start(at_load, &plan.scales);
            let mut scale_word = 0u32;
            #[unroll]
            for c in 0..scales_a_load {
                let origin =
                    Self::block_origin(&scales_start, comptime!(c as u32).runtime(), plan.clone());
                let mut lines = Array::<Vector<F, SW>>::new(src_lines);
                let mut most = F::new(0.0);
                #[unroll]
                for l in 0..src_lines {
                    let held = Self::read_line::<F, SW>(
                        &read,
                        &origin,
                        comptime!(l as u32).runtime(),
                        plan.clone(),
                    );
                    lines[l] = held;
                    most = max(most, Self::largest::<F, SW>(held));
                }
                let inner = Self::block_scale::<F>(most, coarse, plan.clone());
                let mut codes = Array::<Vector<u32, SW>>::new(src_lines);
                #[unroll]
                for l in 0..src_lines {
                    codes[l] =
                        Self::encode_line::<F, SW>(lines[l], most, inner, coarse, plan.clone());
                }
                #[unroll]
                for j in 0..loads_a_block {
                    let at = Self::block_load(
                        &origin,
                        comptime!(j as u32).runtime(),
                        comptime!(plan.values.clone()),
                        comptime!(plan.block.clone()),
                        comptime!(plan.load.clone()),
                    );
                    let mut load_words = Vector::<u32, WP>::new(0u32);
                    #[unroll]
                    for l in 0..comptime!(plan.lines_a_load()) {
                        load_words |= Self::place_codes::<SW, WP>(
                            codes[comptime!(j * plan.lines_a_load() + l)],
                            comptime!(l as u32).runtime(),
                            plan.clone(),
                        );
                    }
                    words.write(
                        comptime!(plan.load.clone()).index(&at, &plan.values),
                        load_words,
                    );
                }
                match comptime!(plan.scale_packing) {
                    Packing::Packed { field } => {
                        let shift = comptime!((c * field.size_bits()) as u32);
                        scale_word |= Self::scale_code::<F>(inner, field) << shift;
                    }
                    Packing::Plain => {
                        Self::write_plain_scale::<F>(scales, inner, &origin, plan.clone())
                    }
                }
            }
            if comptime!(matches!(plan.scale_packing, Packing::Packed { .. })) {
                let mut scale_words = scales.nd_words_mut::<Const<1>>();
                scale_words.write(
                    comptime!(plan.scale_load.clone()).index(&scales_start, &plan.scales),
                    Vector::new(scale_word),
                );
            }
        }
    }

    /// The plane's lanes taking the window together, a line of the source a lane: a block's lanes
    /// meet its largest magnitude through the plane, each encodes its own line, a written load's
    /// lanes gather their codes into it, and a load of scales' lanes gather their blocks' scales
    /// into it. Every lane takes as many turns as the next, so each meeting is the whole plane's;
    /// a lane past the window's last line takes part with nothing and writes nothing.
    fn quantize_across_lanes<F: Float>(
        &mut self,
        src: &Tile,
        scales: &mut Tile,
        coarse: f32,
        #[comptime] fill: FillUnits,
        #[comptime] plan: EncodingCopy,
    ) {
        let size!(SW) = comptime!(plan.line.values());
        let size!(WP) = comptime!(plan.packing.physical(plan.load.values()));
        let block_lanes = comptime!(plan.lines_a_block() as u32);
        let load_lanes = comptime!(plan.lines_a_load() as u32);
        let scale_lanes = comptime!(plan.lines_a_scale_load() as u32);
        let lines = comptime!((plan.scale_loads() * plan.lines_a_scale_load()) as u32).runtime();
        let read = src.nd_packed::<F, SW>(comptime!(Guard::Checked));
        let mut words = self.nd_words_mut::<WP>();
        let lane = FillUnits::worker(fill) as u32;
        let width = FillUnits::workers(fill) as u32;
        let turns = lines.div_ceil(width);
        for turn in 0..turns {
            let at_line = turn * width + lane;
            let held_line = at_line < lines;
            // A lane past the window reads the window's first line, and keeps nothing of it.
            let at_line = select(held_line, at_line, 0u32);
            let at_load = at_line / scale_lanes;
            let in_block = at_line % block_lanes;
            let c = (at_line % scale_lanes) / block_lanes;
            let scales_start = comptime!(plan.scale_load.clone()).start(at_load, &plan.scales);
            let origin = Self::block_origin(&scales_start, c, plan.clone());
            let line = Self::read_line::<F, SW>(&read, &origin, in_block, plan.clone());
            let mut most = select(held_line, Self::largest::<F, SW>(line), F::new(0.0));
            #[unroll]
            for step in 0..comptime!(block_lanes.trailing_zeros()) {
                most = max(most, plane_shuffle_xor(most, comptime!(1u32 << step)));
            }
            let inner = Self::block_scale::<F>(most, coarse, plan.clone());
            let codes = Self::encode_line::<F, SW>(line, most, inner, coarse, plan.clone());
            // This lane's codes in its place in the load, then the load's lanes' together.
            let in_load = in_block % load_lanes;
            let mut load_words = Self::place_codes::<SW, WP>(codes, in_load, plan.clone());
            #[unroll]
            for step in 0..comptime!(load_lanes.trailing_zeros()) {
                load_words = load_words | plane_shuffle_xor(load_words, comptime!(1u32 << step));
            }
            if held_line && in_load == 0 {
                let at = Self::block_load(
                    &origin,
                    in_block,
                    comptime!(plan.values.clone()),
                    comptime!(plan.block.clone()),
                    comptime!(plan.line.clone()),
                );
                words.write(
                    comptime!(plan.load.clone()).index(&at, &plan.values),
                    load_words,
                );
            }
            match comptime!(plan.scale_packing) {
                Packing::Packed { field } => {
                    let bits = comptime!(field.size_bits() as u32);
                    let code = Self::scale_code::<F>(inner, field) << (c * bits);
                    let mut scale_word = select(in_block == 0, code, 0u32);
                    #[unroll]
                    for step in 0..comptime!(scale_lanes.trailing_zeros()) {
                        scale_word |= plane_shuffle_xor(scale_word, comptime!(1u32 << step));
                    }
                    if held_line && at_line.is_multiple_of(scale_lanes) {
                        let mut scale_words = scales.nd_words_mut::<Const<1>>();
                        scale_words.write(
                            comptime!(plan.scale_load.clone()).index(&scales_start, &plan.scales),
                            Vector::new(scale_word),
                        );
                    }
                }
                Packing::Plain => {
                    if held_line && in_block == 0 {
                        Self::write_plain_scale::<F>(scales, inner, &origin, plan.clone());
                    }
                }
            }
        }
    }

    /// The first value of the `c`-th block of the load of scales starting at `start`: its scale's
    /// coordinates, `c` further along the axis the load runs along, and zero along the axes the
    /// scales omit.
    fn block_origin(start: &Coords<u32>, c: u32, #[comptime] plan: EncodingCopy) -> Coords<u32> {
        let (along, _) = comptime!(plan.scale_load.along_one_axis("Tile::quantize_from"));
        let mut origin = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(plan.values.rank()) {
            let axis = comptime!(plan.values.axis_at(p));
            if comptime!(plan.scales.contains(axis)) {
                let at = start.at(comptime!(plan.scales.position(axis)));
                if comptime!(axis == along) {
                    origin.push(at + c);
                } else {
                    origin.push(at);
                }
            } else {
                origin.push(0u32);
            }
        }
        origin
    }

    /// The `l`-th line of the source's block starting at `origin`.
    fn read_line<F: Float, SW: Size>(
        read: &Masked<'_, Vector<F, SW>, CoordsDyn>,
        origin: &Coords<u32>,
        l: u32,
        #[comptime] plan: EncodingCopy,
    ) -> Vector<F, SW> {
        let at = Self::block_load(
            origin,
            l,
            comptime!(plan.values.clone()),
            comptime!(plan.block.clone()),
            comptime!(plan.line.clone()),
        );
        let at = Self::coords_over(
            &at,
            comptime!(plan.values.clone()),
            comptime!(plan.source.clone()),
        );
        read.read(comptime!(plan.line.clone()).index(&at, &plan.source))
    }

    /// The largest magnitude of `line`.
    fn largest<F: Float, SW: Size>(line: Vector<F, SW>) -> F {
        let mut most = F::new(0.0);
        #[unroll]
        for e in 0..SW::value() {
            most = max(most, F::abs(line.extract(e)));
        }
        most
    }

    /// A block's scale, its largest magnitude `most`, as burn calibrates it: the block's range,
    /// twice its largest, over the format's, in the source's element, rounded up to the scale's
    /// dtype, which never clips the block's largest. Spelled as burn's ops: a device's fast
    /// division rounds `2a / (b - a)` and `a / ((b - a) / 2)` apart. Under a coarser scale,
    /// `coarse`, the block's is taken relative to it, in `f32`.
    fn block_scale<F: Float>(most: F, coarse: f32, #[comptime] plan: EncodingCopy) -> F {
        let (low, high) = comptime!(plan.value.range());
        let wanted = most * F::new(2.0) / F::new(comptime!((high - low) as f64));
        if comptime!(plan.two_levels) {
            round_up_to_dtype::<F>(F::cast_from(f32::cast_from(wanted) / coarse), plan.dtype)
        } else {
            round_up_to_dtype::<F>(wanted, plan.dtype)
        }
    }

    /// `line`'s codes under its block's scale `inner`, beneath `coarse` where there is one. An
    /// all-zero block, whose largest magnitude `most` is zero, has a zero scale, which no value
    /// is over: its codes are zero.
    fn encode_line<F: Float, SW: Size>(
        line: Vector<F, SW>,
        most: F,
        inner: F,
        coarse: f32,
        #[comptime] plan: EncodingCopy,
    ) -> Vector<u32, SW> {
        let scaled = if comptime!(plan.two_levels) {
            Vector::<f32, SW>::cast_from(line) / Vector::new(coarse * f32::cast_from(inner))
        } else {
            Vector::<f32, SW>::cast_from(line / Vector::new(inner))
        };
        let encoded = Self::encode_values::<SW>(scaled, plan.value);
        select(most == F::new(0.0), Vector::new(0u32), encoded)
    }

    /// A written load's words holding `codes`, a line's, as its `in_load`-th line: each word's
    /// first code in its low bits, the other lines' fields zero.
    fn place_codes<SW: Size, WP: Size>(
        codes: Vector<u32, SW>,
        in_load: u32,
        #[comptime] plan: EncodingCopy,
    ) -> Vector<u32, WP> {
        let bits = comptime!(plan.value.size_bits());
        let per_word = comptime!(32 / bits);
        let sw = comptime!(plan.line.values());
        let mask = comptime!(((1u64 << bits) - 1) as u32);
        let mut words = Vector::<u32, WP>::empty();
        #[unroll]
        for w in 0..WP::value() {
            let mut word = 0u32;
            #[unroll]
            for i in 0..per_word {
                let at = comptime!(w * per_word + i);
                let field =
                    (codes.extract(comptime!(at % sw)) & mask) << comptime!((i * bits) as u32);
                word |= select(in_load == comptime!((at / sw) as u32), field, 0u32);
            }
            words.insert(w, word);
        }
        words
    }

    /// The first value of the `j`-th of a block's loads, the block starting at `origin`: the
    /// loads walk the block's space as they walk any window's, the last axis fastest.
    fn block_load(
        origin: &Coords<u32>,
        j: u32,
        #[comptime] values: Space,
        #[comptime] block: Space,
        #[comptime] load: VectorTile,
    ) -> Coords<u32> {
        let inside = load.start(j, &block);
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

    /// `at`, coordinates over `from`, restated over `to`, whose axes are all `from`'s: the
    /// source carries no axis the values span alone, such as the blocks of an axis no scale
    /// groups, which is one block.
    fn coords_over(
        at: &Coords<u32>,
        #[comptime] from: Space,
        #[comptime] to: Space,
    ) -> Coords<u32> {
        let mut over = Coords::<u32>::new();
        #[unroll]
        for p in 0..comptime!(to.rank()) {
            over.push(at.at(comptime!(from.position(to.axis_at(p)))));
        }
        over
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

    /// `scale`'s code as `field` stores it in a word, in the low bits. Exact: the scale is already
    /// rounded onto its dtype's grid.
    fn scale_code<F: Float>(scale: F, #[comptime] field: Field) -> u32 {
        let rounded = Vector::<f32, Const<1>>::new(f32::cast_from(scale));
        let code = match comptime!(field.scale_dtype()) {
            ScaleDtype::UE8M0 => f32_to_ue8m0_bits::<Const<1>>(rounded),
            ScaleDtype::UE4M3 => f32_to_fp8_bits::<Const<1>>(rounded, Fp8Format::E4M3),
            ScaleDtype::F32 => Vector::<u32, Const<1>>::reinterpret(rounded),
            ScaleDtype::F16 => Vector::new(u32::cast_from(u16::reinterpret(half::f16::cast_from(
                rounded.extract(0usize),
            )))),
            // `bf16` is a truncated `f32`, and the scale already lies on its grid.
            ScaleDtype::BF16 => Vector::<u32, Const<1>>::reinterpret(rounded) >> Vector::new(16u32),
        };
        code.extract(0usize)
    }

    /// `scale` written into `scales`, stored as plain floats, at the block starting at `origin`.
    fn write_plain_scale<F: Float>(
        scales: &mut Tile,
        scale: F,
        origin: &Coords<u32>,
        #[comptime] plan: EncodingCopy,
    ) {
        let define!(Stored) = scales.element;
        let mut out = scales.nd_mut::<Stored, Const<1>>();
        let at = Self::coords_over(
            origin,
            comptime!(plan.values.clone()),
            comptime!(plan.scales.clone()),
        );
        out.write(
            comptime!(plan.scale_load.clone()).index(&at, &plan.scales),
            Vector::new(Stored::cast_from(scale)),
        );
    }
}
