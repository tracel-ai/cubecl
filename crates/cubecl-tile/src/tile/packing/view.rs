//! The unpacking read: a stored `u32`'s fields served as values.

use std::marker::PhantomData;

use cubecl::e2m1x2;
use cubecl::ir::FloatKind;
use cubecl::ir::VectorSize;
use cubecl::ir::types::Fp8Format;
use cubecl::post_processing::fp4::{e2m1_words_to_f16, e2m1_words_to_f16_placed};
use cubecl::post_processing::minifloat::{fp8_bits_to_f32, ue8m0_bits_to_f32};
use cubecl::prelude::barrier::Barrier;
use cubecl::quant::scheme::QuantValue;
use cubecl_std::quant::fp4::e2m1_packed_bits_to_float;

use crate::{Field, FieldDecode, Packing};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    View, ViewExpand, ViewOperations, ViewOperationsExpand, layout::Coordinates,
};

/// Unpack one line of `NQ` stored words into its `NF` values, fields from each word's low bits.
#[cube]
pub(crate) fn unpack_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: Field,
) -> Vector<F, NF> {
    match comptime!(field.decode()) {
        FieldDecode::SignExtended => unpack_int_line::<F, NQ, NF>(words, field),
        FieldDecode::Unsigned => unpack_index_line::<F, NQ, NF>(words, field),
        FieldDecode::Reinterpreted => unpack_fp4_line::<F, NQ, NF>(words),
        FieldDecode::Converted => {
            unpack_e2m1_line::<F, NQ, NF>(words, comptime!(FieldDecode::Converted))
        }
        FieldDecode::Placed { lifted } => {
            unpack_e2m1_line::<F, NQ, NF>(words, comptime!(FieldDecode::Placed { lifted }))
        }
        FieldDecode::Byte(format) => unpack_byte_line::<F, NQ, NF>(words, format),
        FieldDecode::Bits(kind) => unpack_float_line::<F, NQ, NF>(words, kind),
    }
}

/// The `NF` values starting at `offset` of one stored line, read as `packing` lays them.
#[cube]
pub(crate) fn values_at<F: Numeric, I: Numeric, WP: Size, NF: Size>(
    line: Vector<I, WP>,
    #[comptime] offset: usize,
    #[comptime] packing: Packing,
) -> Vector<F, NF> {
    let nf = NF::value();
    let wp = WP::value();
    match comptime!(packing) {
        Packing::Plain if comptime!(offset == 0 && wp == nf) => Vector::<F, NF>::cast_from(line),
        Packing::Plain => {
            let mut out = Vector::<F, NF>::empty();
            #[unroll]
            for j in 0..nf {
                out.insert(j, F::cast_from(line.extract(offset + j)));
            }
            out
        }
        Packing::Packed { field } => {
            let (bits, per_word) = comptime!((field.size_bits(), field.per_word()));
            let nq = comptime!((nf / per_word).max(1));
            let size!(NQ) = nq;
            let (first, shift) =
                comptime!((offset / per_word, ((offset % per_word) * bits) as u32));
            let mut words = Vector::<u32, NQ>::empty();
            #[unroll]
            for w in 0..nq {
                words.insert(w, u32::cast_from(line.extract(first + w)) >> shift);
            }
            unpack_line::<F, NQ, NF>(words, field)
        }
    }
}

/// Fields one word contributes to an `nf`-wide line of `nq` words.
fn fields_per_word(nq: usize, nf: usize, bits: usize) -> usize {
    let per_word = u32::BITS as usize / bits;
    let fields = nf / nq;
    assert!(
        fields > 0 && fields <= per_word && nf == fields * nq,
        "unpack_line: {nf} values from {nq} words of {per_word} fields"
    );
    fields
}

/// The integer fields, sign-extended out of their slots.
#[cube]
fn unpack_int_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: Field,
) -> Vector<F, NF> {
    let bits = comptime!(field.size_bits());
    let nq = NQ::value();
    let nf = NF::value();
    let factor = comptime!(fields_per_word(nq, nf, bits));
    let mask = comptime!(((1u64 << bits) - 1) as u32);
    let sign = comptime!(1u32 << (bits - 1));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * factor;
        #[unroll]
        for j in 0..factor {
            let raw = (word >> comptime!((j * bits) as u32)) & mask;
            // Branchless sign extension: `(raw ^ s) - s` with `s = 2^(bits-1)`.
            let value = (raw ^ sign) as i32 - sign as i32;
            out.insert(base + j, F::cast_from(value));
        }
    }
    out
}

/// The index fields, their raw bits as they lie.
#[cube]
fn unpack_index_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] field: Field,
) -> Vector<F, NF> {
    let bits = comptime!(field.size_bits());
    let nq = NQ::value();
    let nf = NF::value();
    let factor = comptime!(fields_per_word(nq, nf, bits));
    let mask = comptime!(((1u64 << bits) - 1) as u32);

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * factor;
        #[unroll]
        for j in 0..factor {
            out.insert(
                base + j,
                F::cast_from((word >> comptime!((j * bits) as u32)) & mask),
            );
        }
    }
    out
}

/// The `e2m1` fields, decoded a byte pair at a time in software, on a device that does not convert
/// `e2m1x2` itself.
#[cube]
fn unpack_fp4_line<F: Numeric, NQ: Size, NF: Size>(words: Vector<u32, NQ>) -> Vector<F, NF> {
    let pair = comptime!(QuantValue::E2M1.native_packing());
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, QuantValue::E2M1.size_bits()));
    let bytes = comptime!(fields.div_ceil(pair));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * fields;
        #[unroll]
        for j in 0..bytes {
            let byte = (word >> comptime!((j * 8) as u32)) & 0xff;
            let values = e2m1_packed_bits_to_float::<F, Const<2>>(byte);
            out.insert(base + j * pair, values.extract(0usize));
            if comptime!(j * pair + 1 < fields) {
                out.insert(base + j * pair + 1, values.extract(1usize));
            }
        }
    }
    out
}

/// The `e2m1` fields, eight a word, `decode` saying how a word becomes them: cast as four
/// `e2m1x2` on a device that converts them, the compiler choosing how; or, on a device that
/// emulates the conversion, decoded from the word's bits as `f16` pairs, lifted to their values or
/// placed, each [`E2M1_F16_LIFT`](cubecl::post_processing::fp4::E2M1_F16_LIFT) short of its value,
/// for a reader whose factor carries the lift. A word whose line takes fewer of its fields keeps
/// the first.
#[cube]
fn unpack_e2m1_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] decode: FieldDecode,
) -> Vector<F, NF> {
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, QuantValue::E2M1.size_bits()));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = Vector::<u32, Const<1>>::new(words.extract(w));
        let values = if comptime!(decode == FieldDecode::Placed { lifted: true }) {
            Vector::<F, Const<8>>::cast_from(e2m1_words_to_f16::<Const<1>, Const<8>>(word))
        } else if comptime!(decode == FieldDecode::Placed { lifted: false }) {
            Vector::<F, Const<8>>::cast_from(e2m1_words_to_f16_placed::<Const<1>, Const<8>>(word))
        } else {
            Vector::<F, Const<8>>::cast_from(Vector::<e2m1x2, Const<4>>::reinterpret(
                words.extract(w),
            ))
        };
        #[unroll]
        for j in 0..fields {
            out.insert(w * fields + j, values.extract(j));
        }
    }
    out
}

/// The 8-bit float codes, one byte each, through the format's decoder on `u32` patterns.
#[cube]
fn unpack_byte_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] format: Fp8Format,
) -> Vector<F, NF> {
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, 8));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * fields;
        #[unroll]
        for j in 0..fields {
            let byte = Vector::<u32, Const<1>>::new((word >> comptime!((j * 8) as u32)) & 0xff);
            let value = match comptime!(format) {
                Fp8Format::UE8M0 => ue8m0_bits_to_f32::<Const<1>>(byte),
                _ => fp8_bits_to_f32::<Const<1>>(byte, format),
            };
            out.insert(base + j, F::cast_from(value.extract(0usize)));
        }
    }
    out
}

/// One `f16` bit pattern decoded to `f32` in integer arithmetic; high bits are ignored.
#[cube]
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits & 0x8000u32) << 16u32;
    let exponent = (bits >> 10u32) & 0x1fu32;
    let mantissa = bits & 0x3ffu32;
    // Rebias the exponent 15 -> 127 and widen the mantissa; a zero exponent is subnormal.
    let normal = sign | ((exponent + 112u32) << 23u32) | (mantissa << 13u32);
    let subnormal = sign | u32::reinterpret(f32::cast_from(mantissa) * 5.9604645e-8f32);
    let finite = select(exponent == 0u32, subnormal, normal);
    let special = select(
        mantissa == 0u32,
        sign | 0x7f80_0000u32,
        sign | 0x7fc0_0000u32,
    );
    f32::reinterpret(select(exponent == 31u32, special, finite))
}

/// The whole floats, each read out of its own slot.
#[cube]
fn unpack_float_line<F: Numeric, NQ: Size, NF: Size>(
    words: Vector<u32, NQ>,
    #[comptime] kind: FloatKind,
) -> Vector<F, NF> {
    let bits = comptime!(Field::float_bits(kind));
    let nq = NQ::value();
    let nf = NF::value();
    let fields = comptime!(fields_per_word(nq, nf, bits));

    let mut out = Vector::<F, NF>::empty();
    #[unroll]
    for w in 0..words.vector_size() {
        let word = words.extract(w);
        let base = w * fields;
        #[unroll]
        for j in 0..fields {
            let slot = word >> comptime!((j * bits) as u32);
            // `bf16` is a truncated `f32`, so shifting it back is the decode.
            let value = match comptime!(kind) {
                FloatKind::F32 => F::cast_from(f32::reinterpret(slot)),
                FloatKind::F16 => F::cast_from(f16_bits_to_f32(slot)),
                _ => F::cast_from(f32::reinterpret((slot & 0xffffu32) << 16u32)),
            };
            out.insert(base + j, value);
        }
    }
    out
}

/// A [`View`] over stored words serving `Vector<F, NF>` lines, `NF = NQ * factor`.
#[expect(dead_code, reason = "read through the expand impls below")]
#[derive(CubeType, Clone)]
pub(crate) struct PackedView<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> {
    words: View<'a, Vector<u32, NQ>, C>,
    #[cube(comptime)]
    field: Field,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> PackedView<'a, NQ, F, NF, C> {
    pub(crate) fn new(words: View<'a, Vector<u32, NQ>, C>, #[comptime] field: Field) -> Self {
        PackedView::<'a, NQ, F, NF, C> {
            words,
            field,
            _ty: PhantomData,
        }
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> PackedView<'a, NQ, F, NF, C> {
    /// This view as a plain [`View`] of served values.
    pub(crate) fn view(self) -> View<'a, Vector<F, NF>, C> {
        unexpanded!()
    }

    pub(crate) fn __expand_view(
        scope: &Scope,
        this: PackedViewExpand<'a, NQ, F, NF, C>,
    ) -> ViewExpand<'a, Vector<F, NF>, C> {
        this.__expand_view_method(scope)
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static>
    PackedViewExpand<'a, NQ, F, NF, C>
{
    /// Binds `NQ` to the width these words are read at, before a read.
    ///
    /// A width is bound by its `size!` site, one marker however many operands the site serves:
    /// two packed operands of a kernel bound at different widths each rebind it, and a view read
    /// after the other was built would otherwise unpack at the other's width.
    fn own_width(&self, scope: &Scope) {
        let words = self.words.__expand_vector_size_method(scope);
        scope.register_size::<NQ>(words);
    }

    fn unpack(
        &self,
        scope: &Scope,
        words: NativeExpand<Vector<u32, NQ>>,
    ) -> NativeExpand<Vector<F, NF>> {
        unpack_line::expand::<F, NQ, NF>(scope, words, self.field)
    }

    pub(crate) fn __expand_view_method(self, scope: &Scope) -> ViewExpand<'a, Vector<F, NF>, C> {
        ViewExpand::new(scope, self)
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> Vectorized
    for PackedView<'a, NQ, F, NF, C>
{
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> VectorizedExpand
    for PackedViewExpand<'a, NQ, F, NF, C>
{
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.words.__expand_vector_size_method(scope)
            * (u32::BITS as usize / self.field.size_bits())
    }
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static> ViewOperations<Vector<F, NF>, C>
    for PackedView<'a, NQ, F, NF, C>
{
}

impl<'a, NQ: Size, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperationsExpand<Vector<F, NF>, C> for PackedViewExpand<'a, NQ, F, NF, C>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        self.own_width(scope);
        let words = self.words.clone().__expand_read_method(scope, pos);
        self.unpack(scope, words)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        self.own_width(scope);
        let words = self.words.clone().__expand_read_checked_method(scope, pos);
        self.unpack(scope, words)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
        mask_value: NativeExpand<Vector<F, NF>>,
    ) -> NativeExpand<Vector<F, NF>> {
        self.own_width(scope);
        let words = self
            .words
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);
        let value = self.unpack(scope, words);
        select::expand::<Vector<F, NF>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        self.own_width(scope);
        let words = self
            .words
            .clone()
            .__expand_read_unchecked_method(scope, pos);
        self.unpack(scope, words)
    }

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: <C>::ExpandType,
        _end: <C>::ExpandType,
    ) -> &SliceExpand<Vector<F, NF>> {
        panic!("PackedView: a packed operand has no raw slice of served values")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <C>::ExpandType {
        self.words.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<bool> {
        self.words.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<F, NF>>,
        _pos: C::ExpandType,
    ) {
        panic!("PackedView: a tensor map cannot unpack on the fly")
    }
}
