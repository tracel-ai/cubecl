//! The scale a tile carries ([`Tile::mul`](crate::Tile::mul)): a scale tile windowed with the
//! values, never applied by the tile itself. The scale tile is a [`Tile`] of its own and may carry
//! a scale in turn, each covering more than the one inside it: NVFP4's values are under their block
//! scales, which are under the per-tensor scale.

use std::sync::Arc;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::ir::{ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::quant::scheme::QuantValue;
use cubecl::unexpanded;
use cubecl_std::tensor::layout::Coords2d;

use crate::stage::pipeline::payload::base::{StageOperand, StageSpec, stage_one};
use crate::*;

/// The scale a tile carries: none, or a scale tile, itself under its own scale or none.
/// What [`Tile::factor`](crate::Tile::factor) takes out of a tile and
/// [`Tile::scaled`](crate::Tile::scaled) puts on one.
#[derive(Clone)]
pub struct Factor;

#[derive(Clone, Default)]
pub struct FactorExpand {
    // `Arc`: a tile holds its scale, and the scale is a tile.
    scale: Option<Arc<TileExpand>>,
    /// Whether a stage keeps these scales beside the values though they are not words of block
    /// scales ([`Tile::scales_kept_beside`](crate::Tile::scales_kept_beside)).
    kept_beside: bool,
}

impl Factor {
    /// No scale: what every operand carries until [`Tile::mul`](crate::Tile::mul).
    pub(crate) fn none() -> Factor {
        unexpanded!()
    }

    /// This factor windowed to `step`, as the values it rides are.
    pub(crate) fn at(&self, _step: &Step) -> Factor {
        unexpanded!()
    }

    /// This factor with every scale that spans `axis` one wide at `coord` along it, as the values
    /// it rides are ([`Tile::placed_at`](crate::Tile::placed_at)).
    #[allow(dead_code)] // Reached through its expand, from [`Tile::indexed_at`](crate::Tile::indexed_at).
    pub(crate) fn placed_at(&self, _axis: Axis, _coord: u32) -> Factor {
        unexpanded!()
    }

    /// This factor with every scale that spans `axis` reading the run `[from, until)` along it,
    /// as the values it rides are ([`Tile::run_within`](crate::Tile::run_within)).
    #[allow(dead_code)] // Reached through its expand, from [`Tile::indexed_at`](crate::Tile::indexed_at).
    pub(crate) fn run_within(&self, _axis: Axis, _from: u32, _until: u32) -> Factor {
        unexpanded!()
    }

    /// Whether the innermost scales vary along `axis`, so a run along it spans scales.
    pub(crate) fn varies_along(&self, _axis: Axis) -> bool {
        unexpanded!()
    }

    /// The scale covering the value at `coords`: the product down the chain, each scale read
    /// under `guard`. [`Guard::Proved`] where the reader proved the value's read in bounds, which
    /// puts the scales covering it in bounds.
    pub(crate) fn at_coords(&self, _coords: &Coords<u32>, _values: Space, _guard: Guard) -> f32 {
        unexpanded!()
    }

    /// The one scale every value shares: the product down the chain, no scale of which varies
    /// along any axis.
    pub(crate) fn whole(&self) -> f32 {
        unexpanded!()
    }

    /// Whether these values carry a scale at all.
    pub(crate) fn scaled(&self) -> comptime_type!(bool) {
        unexpanded!()
    }

    /// Whether the innermost scales are [words](FactorExpand::holds_words) of block scales.
    pub(crate) fn holds_words(&self) -> comptime_type!(bool) {
        unexpanded!()
    }

    /// The innermost scale tile, without the scale it carries in turn: the block scales an
    /// encoding copy writes ([`Tile::quantize_from`](crate::Tile::quantize_from)).
    pub(crate) fn innermost_scale(&self) -> Tile {
        unexpanded!()
    }

    /// Every scale outside the innermost, multiplied into one value read at the region's origin.
    pub(crate) fn coarse(&self) -> f32 {
        unexpanded!()
    }

    /// How many scales the chain holds.
    pub(crate) fn depth(&self) -> comptime_type!(usize) {
        unexpanded!()
    }

    /// How the innermost scales sit in their buffer: `None` where these values carry no scale.
    pub(crate) fn inner_stored(&self) -> comptime_type!(Option<Packing>) {
        unexpanded!()
    }

    /// The innermost scales' word at the value at `coords`, read as it is stored.
    pub(crate) fn word_at(&self, _coords: &Coords<u32>, _values: Space) -> u32 {
        unexpanded!()
    }

    /// The innermost scales' load holding the scale covering the value at `coords` written as
    /// `scales`, the mirror of [`at_coords`](Factor::at_coords): stored plain, or packed as their
    /// field, as many to a word as it holds (`ue8m0` and `ue4m3` four, `f16` and `bf16` two,
    /// `f32` one). A word is written whole, so `scales` is its every scale.
    #[allow(dead_code)] // Reached through its expand, from [`Tile::quantize_from`](crate::Tile::quantize_from).
    pub(crate) fn write_at<N: Size>(
        &self,
        _coords: &Coords<u32>,
        _values: Space,
        _scales: Vector<f32, N>,
    ) {
        unexpanded!()
    }

    /// The innermost scales calibrated from `src`, the float values they scale, as burn's
    /// `quantize_dynamic` calibrates them for values written as `value`: each block's largest
    /// magnitude over the format's, relative to `coarse`, every scale outside the innermost, where
    /// there is one. The units that fill the scales take their blocks in turn: lanes a block each,
    /// meeting through the plane where a word packs several; otherwise a unit a word at a time
    /// ([`Tile::quantize_from`](crate::Tile::quantize_from)).
    #[allow(dead_code)] // Reached through its expand, from [`Tile::quantize_from`](crate::Tile::quantize_from).
    #[allow(clippy::extra_unused_type_parameters)] // `F` is the source's float, which the expand reads.
    pub(crate) fn calibrate_from<F: Float>(&self, _src: &Tile, _coarse: f32, _value: QuantValue) {
        unexpanded!()
    }

    pub(crate) fn __expand_none(_scope: &Scope) -> FactorExpand {
        FactorExpand::default()
    }
}

// A scale is expand-time data, held by one thread: `Arc` as the erased tensors hold theirs.
#[allow(clippy::arc_with_non_send_sync)]
impl FactorExpand {
    /// A factor of `scale`, read at the values' coordinates, under whatever scale it carries.
    pub(crate) fn of(scale: &TileExpand) -> Self {
        FactorExpand {
            scale: Some(Arc::new(scale.clone())),
            kept_beside: false,
        }
    }

    /// This factor, its scales kept beside the values by a stage as `from`'s are: a factor
    /// windowed, split or staged off another keeps its choice.
    fn keeping(self, from: &FactorExpand) -> Self {
        FactorExpand {
            kept_beside: from.kept_beside,
            ..self
        }
    }

    /// This factor, its scales kept beside the values by a stage.
    pub(crate) fn kept_beside(self) -> Self {
        FactorExpand {
            kept_beside: true,
            ..self
        }
    }

    /// The innermost scale tile, without the scale it carries in turn.
    pub(crate) fn innermost_tile(&self) -> Option<TileExpand> {
        self.scale.as_deref().map(|scale| {
            let mut alone = scale.clone();
            alone.factor = FactorExpand::default();
            alone
        })
    }

    /// The scale tiles down the chain, innermost first.
    fn chain(&self) -> impl Iterator<Item = &TileExpand> {
        std::iter::successors(self.scale.as_deref(), |scale| scale.factor.scale.as_deref())
    }

    /// Whether `scale`'s own scales vary along an axis outside `space`.
    fn varies_outside(scope: &Scope, scale: &TileExpand, space: &Space) -> bool {
        let projection = scale.clone().__expand_projection_method(scope);
        scale
            .place
            .space
            .axes()
            .any(|axis| projection.addresses(axis) && !space.contains(axis))
    }

    /// This factor cut in two: the inner scales, down to the last one that varies along an axis
    /// outside `space`, and the rest, which vary only along `space`. Each scale covers more than
    /// the one inside it, so once one varies only along `space`, every one outside it does too.
    pub(crate) fn split_on(&self, scope: &Scope, space: &Space) -> (FactorExpand, FactorExpand) {
        let Some(scale) = self.scale.as_deref() else {
            return (FactorExpand::default(), FactorExpand::default());
        };
        if !Self::varies_outside(scope, scale, space) {
            return (FactorExpand::default().keeping(self), self.clone());
        }
        let (kept, taken) = scale.factor.split_on(scope, space);
        let mut inner = scale.clone();
        inner.factor = kept;
        (FactorExpand::of(&inner).keeping(self), taken)
    }

    /// This factor with `coarser` outside it: under the last scale of its chain.
    pub(crate) fn and(&self, coarser: FactorExpand) -> FactorExpand {
        match self.scale.as_deref() {
            None => coarser,
            Some(scale) => {
                let mut outer = scale.clone();
                outer.factor = scale.factor.and(coarser);
                FactorExpand::of(&outer)
            }
        }
    }

    /// Whether these values carry a scale at all.
    pub(crate) fn scaled(&self) -> bool {
        self.scale.is_some()
    }

    /// How many scales the chain holds.
    pub(crate) fn depth(&self) -> usize {
        self.chain().count()
    }

    /// This factor's innermost scale alone.
    pub(crate) fn innermost(&self) -> FactorExpand {
        match self.innermost_tile() {
            Some(scale) => FactorExpand::of(&scale),
            None => FactorExpand::default(),
        }
    }

    /// Every scale outside the innermost, multiplied into one value read at the region's origin.
    pub(crate) fn coarse(&self, scope: &Scope) -> NativeExpand<f32> {
        let mut scaled: NativeExpand<f32> =
            ExpandValue::constant(1u64.into(), f32::elem_type(scope)).into();
        for scale in self.chain().skip(1) {
            let origin = origin_of(scope, &scale.place.space);
            let one = scale_for(scope, scale, &origin, &scale.place.space, Guard::Checked);
            scaled = MulExpand::__expand_mul_method(scaled, scope, one);
        }
        scaled
    }

    pub(crate) fn __expand_at_method(&self, scope: &Scope, step: &StepExpand) -> FactorExpand {
        match self.scale.as_deref() {
            // The scale's own scale is windowed with it, by its `at_step`.
            Some(scale) => {
                FactorExpand::of(&scale.clone().__expand_at_step_method(scope, step)).keeping(self)
            }
            None => FactorExpand::default(),
        }
    }

    pub(crate) fn __expand_placed_at_method(
        &self,
        scope: &Scope,
        axis: Axis,
        coord: NativeExpand<u32>,
    ) -> FactorExpand {
        self.spanning(axis, |scale| {
            scale.__expand_placed_at_method(scope, axis, coord)
        })
    }

    pub(crate) fn __expand_run_within_method(
        &self,
        scope: &Scope,
        axis: Axis,
        from: NativeExpand<u32>,
        until: NativeExpand<u32>,
    ) -> FactorExpand {
        self.spanning(axis, |scale| {
            scale.__expand_run_within_method(scope, axis, from, until)
        })
    }

    /// This factor with `place` applied to every scale that spans `axis`, which carries its own
    /// scales along; the scales that do not span it, as they are.
    pub(crate) fn spanning(
        &self,
        axis: Axis,
        place: impl Fn(&TileExpand) -> TileExpand,
    ) -> FactorExpand {
        match self.scale.as_deref() {
            None => FactorExpand::default(),
            Some(scale) if scale.place.space.contains(axis) => {
                FactorExpand::of(&place(scale)).keeping(self)
            }
            Some(scale) => {
                let mut kept = scale.clone();
                kept.factor = scale.factor.spanning(axis, place);
                FactorExpand::of(&kept).keeping(self)
            }
        }
    }

    pub(crate) fn __expand_at_coords_method(
        &self,
        scope: &Scope,
        coords: &CoordsExpand<u32>,
        values: Space,
        guard: Guard,
    ) -> NativeExpand<f32> {
        let mut product: Option<NativeExpand<f32>> = None;
        for scale in self.chain() {
            assert!(
                !holds_words(scope, scale),
                "Factor: these scales are words of four e4m3 block scales, which only the \
                 block-scaled instruction reads; widened one at a time they mean nothing"
            );
            let one = scale_for(scope, scale, coords, &values, guard);
            product = Some(match product {
                None => one,
                Some(product) => MulExpand::__expand_mul_method(product, scope, one),
            });
        }
        product.unwrap_or_else(|| ExpandValue::constant(1u64.into(), f32::elem_type(scope)).into())
    }

    pub(crate) fn __expand_whole_method(&self, scope: &Scope) -> NativeExpand<f32> {
        let mut product: NativeExpand<f32> =
            ExpandValue::constant(1u64.into(), f32::elem_type(scope)).into();
        for scale in self.chain() {
            let projection = scale.clone().__expand_projection_method(scope);
            assert!(
                scale
                    .place
                    .space
                    .axes()
                    .all(|axis| !projection.addresses(axis)),
                "Factor::whole: a scale that varies across the output is applied where the output \
                 is written through memory, not on the sum's fragments"
            );
            let origin = origin_of(scope, &scale.place.space);
            let one = scale_for(scope, scale, &origin, &scale.place.space, Guard::Checked);
            product = MulExpand::__expand_mul_method(product, scope, one);
        }
        product
    }

    pub(crate) fn __expand_innermost_scale_method(&self, _scope: &Scope) -> TileExpand {
        self.innermost_tile()
            .expect("Tile::quantize_from: the values are written under the block scales they carry")
    }

    pub(crate) fn __expand_coarse_method(&self, scope: &Scope) -> NativeExpand<f32> {
        self.coarse(scope)
    }

    pub(crate) fn __expand_depth_method(&self, _scope: &Scope) -> usize {
        self.depth()
    }

    pub(crate) fn __expand_inner_stored_method(&self, scope: &Scope) -> Option<Packing> {
        self.scale
            .as_deref()
            .map(|scale| scale.clone().__expand_packing_method(scope))
    }

    pub(crate) fn __expand_scaled_method(&self, _scope: &Scope) -> bool {
        self.scaled()
    }

    pub(crate) fn __expand_holds_words_method(&self, scope: &Scope) -> bool {
        self.scale
            .as_deref()
            .is_some_and(|scale| holds_words(scope, scale))
    }

    /// Whether these scales are staged beside the values they scale rather than decoded into
    /// them, so the values under them are staged as they are stored: words of block scales, which
    /// only the block-scaled instruction reads, or scales a reader that decodes as it loads asked
    /// to keep there ([`Tile::scales_kept_beside`](crate::Tile::scales_kept_beside)).
    pub(crate) fn staged_beside(&self, scope: &Scope) -> bool {
        self.scaled() && (self.kept_beside || self.__expand_holds_words_method(scope))
    }

    /// The scale tile staged beside the values, which are under it alone.
    fn beside(&self) -> &TileExpand {
        let scale = self
            .scale
            .as_deref()
            .expect("Factor: these values carry no scale to stage beside them");
        assert!(
            !scale.factor.scaled(),
            "Factor: values under words of block scales carry no other scale, since the \
             block-scaled instruction that reads them holds one; take the coarser one out for the \
             sum (`Tile::factor`)"
        );
        scale
    }

    /// This factor's stage as `spec` shapes the values': the scales staged beside them, laid out
    /// plain whatever the values' storage, since no instruction fragment-loads them.
    pub(crate) fn staged(&self, scope: &Scope, spec: &StageSpec) -> FactorExpand {
        let spec = StageSpec {
            storage: StageStorage::Strided,
            width: None,
            ..spec.clone()
        };
        FactorExpand::of(&stage_one::expand(scope, self.beside(), spec)).keeping(self)
    }

    /// Fill this factor's stage from `src`, the factor it stages windowed as the values are.
    pub(crate) fn fill_from(&self, scope: &Scope, src: &FactorExpand, meeting: &MeetingExpand) {
        meeting.__expand_fill_method(scope, &mut self.beside().clone(), src.beside());
    }

    /// The scales as a slot plans them, beside the values they scale.
    pub(crate) fn stage_operands(&self, scope: &Scope) -> Vec<StageOperand> {
        let scale = self.beside();
        vec![StageOperand {
            delivery: scale.clone().__expand_delivery_method(scope),
            space: scale.place.space.clone(),
            keys: Vec::new(),
        }]
    }

    /// Free the shared memory this factor's stage holds.
    pub(crate) fn free_stage(&self, scope: &Scope) {
        self.beside().clone().__expand_free_stage_method(scope);
    }

    pub(crate) fn __expand_word_at_method(
        &self,
        scope: &Scope,
        coords: &CoordsExpand<u32>,
        values: Space,
    ) -> NativeExpand<u32> {
        let scale = self
            .scale
            .as_deref()
            .filter(|scale| holds_words(scope, scale))
            .expect("Factor::word_at: the innermost scales are no words of block scales");
        scale
            .clone()
            .__expand_scale_for_method::<u32>(scope, coords, values, Guard::Checked)
    }

    pub(crate) fn __expand_write_at_method<N: Size>(
        &self,
        scope: &Scope,
        coords: &CoordsExpand<u32>,
        values: Space,
        scales: NativeExpand<Vector<f32, N>>,
    ) {
        let mut scale = self
            .innermost_tile()
            .expect("Factor::write_at: these values carry no scale to write");
        scale.__expand_write_scales_for_method::<N>(scope, coords, values, scales);
    }

    pub(crate) fn __expand_calibrate_from_method<F: Float>(
        &self,
        scope: &Scope,
        src: &TileExpand,
        coarse: NativeExpand<f32>,
        value: QuantValue,
    ) {
        let mut scale = self
            .innermost_tile()
            .expect("Factor::calibrate_from: these values carry no scale to calibrate");
        scale.__expand_calibrate_method::<F>(scope, src, coarse, value, self.depth() > 1);
    }
}

#[cube]
impl Tile {
    /// The load of these scales holding the one covering the value at `coords` of a tile spanning
    /// `values` written as `scales`, the mirror of [`scale_for`](Tile::scale_for)
    /// ([`Factor::write_at`]).
    #[allow(dead_code)] // Reached through its expand, from [`Factor::write_at`].
    pub(crate) fn write_scales_for<N: Size>(
        &mut self,
        coords: &Coords<u32>,
        #[comptime] values: Space,
        scales: Vector<f32, N>,
    ) {
        let own = self.scale_coords(coords, values);
        self.write_scales::<N>(&own, scales);
    }

    /// This tile's load of scales starting at `start` written as `scales`: plain, or packed into
    /// words as their field.
    pub(crate) fn write_scales<N: Size>(&mut self, start: &Coords<u32>, scales: Vector<f32, N>) {
        let space = comptime!(self.place.space.clone());
        let load = self.vector_tile();
        let n = N::value();
        comptime!(assert!(
            load.values() == n,
            "Factor::write_at: a load of these scales holds {} of them, not {n}",
            load.values()
        ));
        let at = load.index(start, &space);
        let packing = self.packing();
        match comptime!(packing) {
            Packing::Plain => {
                let define!(Stored) = self.element;
                let mut out = self.nd_mut::<Stored, N>();
                out.write(at, Vector::<Stored, N>::cast_from(scales));
            }
            Packing::Packed { field } => {
                let size!(WP) = comptime!(packing.physical(n));
                let mut words = self.nd_words_mut::<WP>();
                words.write(at, pack_line::<WP, N>(scales, field));
            }
        }
    }
}

/// Whether `scale` holds words rather than scales: each `u32` the `e4m3` scales of the four blocks
/// one block-scaled instruction step covers, the first block's in the low byte, as an NVFP4
/// checkpoint packs them, bound plain. Only that instruction reads such scales
/// ([`FactorExpand::__expand_word_at_method`]); widened to a scale one means nothing. A packed
/// scale holds `u32` words too, of fields read as scales.
fn holds_words(scope: &Scope, scale: &TileExpand) -> bool {
    scale.element == u32::elem_type(scope) && scale.__expand_packing_method(scope) == Packing::Plain
}

/// The scale of `scale` covering the value at `coords` of a tile spanning `values`, widened to
/// `f32`, read under `guard`.
fn scale_for(
    scope: &Scope,
    scale: &TileExpand,
    coords: &CoordsExpand<u32>,
    values: &Space,
    guard: Guard,
) -> NativeExpand<f32> {
    scale
        .clone()
        .__expand_scale_for_method::<f32>(scope, coords, values.clone(), guard)
}

/// The all-zero coordinate of `space`: where a scale covering a whole region is read.
fn origin_of(scope: &Scope, space: &Space) -> CoordsExpand<u32> {
    let mut origin = Coords::<u32>::__expand_new(scope);
    for _ in 0..space.rank() {
        let zero: NativeExpand<u32> =
            ExpandValue::constant(0u64.into(), u32::elem_type(scope)).into();
        origin.__expand_push_method(scope, zero);
    }
    origin
}

/// A factor as a leaf reads it: the innermost level per line, coarser levels as one value.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct FactorReader {
    /// The innermost level, or nothing at all.
    pub(crate) inner: Factor,
    /// Every coarser level, already met.
    pub(crate) coarse: f32,
    /// The values' space, which a line's `(row, col)` resolves to a coordinate through.
    #[cube(comptime)]
    pub(crate) values: Space,
    #[cube(comptime)]
    pub(crate) axes: MatrixAxes,
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Whether there is a scale at all.
    #[cube(comptime)]
    pub(crate) scaled: bool,
    /// Which batch matrix the lines are read from.
    pub(crate) matrix: usize,
}

#[cube]
impl FactorReader {
    /// `value`, the line at `pos` of the values' matrix, under the scale covering it.
    pub(crate) fn apply<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        pos: Coords2d,
    ) -> Vector<E, V> {
        if comptime!(self.scaled) {
            let (row, col) = pos;
            let coords = TileMatrix::value_coords(
                row,
                col,
                self.matrix,
                comptime!(&self.values),
                comptime!(self.axes),
                comptime!(self.vector_size),
            );
            self.apply_at::<E, V>(value, &coords)
        } else {
            value
        }
    }

    /// `value`, the line whose first value lies at `coords`, under the scale covering it.
    pub(crate) fn apply_at<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        coords: &Coords<u32>,
    ) -> Vector<E, V> {
        if comptime!(self.scaled) {
            let scale = self.inner.at_coords(
                coords,
                comptime!(self.values.clone()),
                comptime!(Guard::Checked),
            );
            // Multiplied in `f32`, rounded to `E` once: a placed `e2m1` value is a lift short, which
            // the factor carries, and the lift times a scale is past a half's range where its
            // product with the value is not.
            let scaled = Vector::<f32, V>::cast_from(value)
                * Vector::<f32, V>::cast_from(scale * self.coarse);
            Vector::<E, V>::cast_from(scaled)
        } else {
            value
        }
    }
}

impl FactorExpand {
    pub(crate) fn __expand_varies_along_method(&self, scope: &Scope, axis: Axis) -> bool {
        self.scale.as_deref().is_some_and(|scale| {
            scale
                .clone()
                .__expand_projection_method(scope)
                .addresses(axis)
        })
    }
}

impl CubeType for Factor {
    type ExpandType = FactorExpand;
}
impl IntoExpand for FactorExpand {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}
impl ExpandTypeClone for FactorExpand {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}
impl IntoMut for FactorExpand {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}
impl CubeDebug for FactorExpand {}
impl AsRefExpand for FactorExpand {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for FactorExpand {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}
