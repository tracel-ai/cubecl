//! Software fp8 conversion, on `u32` bit patterns only so that a backend with no 8- or 16-bit types
//! can still call it on the bits in a word.

use cubecl_ir::{
    NamedRewrite, Scope,
    dialect::{
        base::OperationPtrExt,
        cmp::{FEqualOp, FNotEqualOp},
        general::CastOp,
    },
    interfaces::TypedExt,
    prelude::*,
    types::Fp8Format,
};
use enumset::EnumSet;
use pliron::r#type::TypeHandle;

use crate::{self as cubecl, prelude::*};

define_size!(N);

const F32_MANTISSA_BITS: u32 = f32::MANTISSA_DIGITS - 1;
const F32_MANTISSA_MASK: u32 = (1 << F32_MANTISSA_BITS) - 1;
const F32_MAGNITUDE_MASK: u32 = u32::MAX >> 1;
const F32_EXPONENT_BIAS: u32 = (f32::MAX_EXP - 1) as u32;
const F32_INFINITY_BITS: u32 = f32::INFINITY.to_bits();
const F32_NAN_BITS: u32 = f32::NAN.to_bits();
const FP8_SIGN_BIT: u32 = 1 << (u8::BITS - 1);
const FP8_MAGNITUDE_MASK: u32 = FP8_SIGN_BIT - 1;
const SIGN_SHIFT: u32 = u32::BITS - u8::BITS;

/// Bits above the low byte are ignored.
#[cube]
pub fn fp8_bits_to_f32<N: Size>(
    bits: Vector<u32, N>,
    #[comptime] format: Fp8Format,
) -> Vector<f32, N> {
    let mantissa_bits = comptime![format.mantissa_bits()];
    let exponent_mask = comptime![(1u32 << format.exponent_bits()) - 1];
    let mantissa_mask = comptime![(1u32 << mantissa_bits) - 1];
    let rebias = comptime![F32_EXPONENT_BIAS - format.bias()];
    let mantissa_shift = comptime![F32_MANTISSA_BITS - mantissa_bits];
    let subnormal_step = comptime![format.subnormal_step()];

    let sign = (bits & Vector::new(FP8_SIGN_BIT)) << Vector::new(SIGN_SHIFT);
    let exponent = (bits >> Vector::new(mantissa_bits)) & Vector::new(exponent_mask);
    let mantissa = bits & Vector::new(mantissa_mask);

    let normal = sign
        | ((exponent + Vector::new(rebias)) << Vector::new(F32_MANTISSA_BITS))
        | (mantissa << Vector::new(mantissa_shift));
    let subnormal = sign
        | Vector::<u32, N>::reinterpret(
            Vector::<f32, N>::cast_from(mantissa) * Vector::new(subnormal_step),
        );
    let value = select_many(exponent.equal(&Vector::new(0u32)), subnormal, normal);

    let nan = sign | Vector::new(F32_NAN_BITS);
    let result = if comptime![format.has_infinity()] {
        let inf = sign | Vector::new(F32_INFINITY_BITS);
        let special = select_many(mantissa.equal(&Vector::new(0u32)), inf, nan);
        select_many(exponent.equal(&Vector::new(exponent_mask)), special, value)
    } else {
        let magnitude = bits & Vector::new(FP8_MAGNITUDE_MASK);
        select_many(
            magnitude.equal(&Vector::new(FP8_MAGNITUDE_MASK)),
            nan,
            value,
        )
    };

    Vector::<f32, N>::reinterpret(result)
}

/// Round to nearest even; overflow and infinities saturate to the largest finite value, as the host
/// codecs do.
#[cube]
pub fn f32_to_fp8_bits<N: Size>(
    value: Vector<f32, N>,
    #[comptime] format: Fp8Format,
) -> Vector<u32, N> {
    let mantissa_bits = comptime![format.mantissa_bits()];
    let mantissa_shift = comptime![F32_MANTISSA_BITS - mantissa_bits];
    let rebias = comptime![format.bias().wrapping_sub(F32_EXPONENT_BIAS)];
    let half_ulp = comptime![1u32 << (mantissa_shift - 1)];
    let subnormal_scale = comptime![1.0 / format.subnormal_step()];
    let min_normal = comptime![format.min_normal()];
    let max_value = comptime![format.max_value()];
    let max_code = comptime![format.max_code()];
    let nan_code = comptime![format.nan_code()];

    let bits = Vector::<u32, N>::reinterpret(value);
    let sign = (bits >> Vector::new(SIGN_SHIFT)) & Vector::new(FP8_SIGN_BIT);
    let magnitude_bits = bits & Vector::new(F32_MAGNITUDE_MASK);
    let magnitude = Vector::<f32, N>::reinterpret(magnitude_bits);

    // Rounding by hand: the usual magic-number trick does not survive fast-math reassociation.
    // `steps` overflows for normal magnitudes, which only feeds the lane the select below
    // discards; no backend traps on a float-to-int overflow.
    let steps = magnitude * Vector::new(subnormal_scale);
    let truncated = Vector::<u32, N>::cast_from(steps);
    let fraction = steps - Vector::<f32, N>::cast_from(truncated);
    let above_half = fraction.greater_than(&Vector::new(0.5f32));
    let tie_to_odd = fraction
        .equal(&Vector::new(0.5f32))
        .vec_and((truncated & Vector::new(1u32)).equal(&Vector::new(1u32)));
    let round_up = Vector::<u32, N>::cast_from(above_half.or(tie_to_odd));
    let subnormal = truncated + round_up;

    let exponent = (magnitude_bits >> Vector::new(F32_MANTISSA_BITS)) + Vector::new(rebias);
    let mantissa = magnitude_bits & Vector::new(F32_MANTISSA_MASK);
    let lsb = (mantissa >> Vector::new(mantissa_shift)) & Vector::new(1u32);
    let rounded =
        ((exponent << Vector::new(F32_MANTISSA_BITS)) | mantissa) + Vector::new(half_ulp - 1) + lsb;
    let normal = rounded >> Vector::new(mantissa_shift);

    let code = select_many(
        magnitude.less_than(&Vector::new(min_normal)),
        subnormal,
        normal,
    );
    let code = select_many(
        magnitude.greater_than(&Vector::new(max_value)),
        Vector::new(max_code),
        code,
    );
    let code = select_many(
        magnitude_bits.greater_than(&Vector::new(F32_INFINITY_BITS)),
        Vector::new(nan_code),
        code,
    );

    code | sign
}

/// How a backend without a native fp8 type holds the bytes of an fp8 vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Fp8Container {
    /// One 8-bit integer per lane.
    #[default]
    Bytes,
    /// Four lanes per `u32`, lane 0 in the low byte, for backends with no 8-bit type at all.
    /// fp8 vectors must then be a multiple of four lanes wide.
    Words,
}

pub type LowerMinifloatCastPass = MatchRewritePass<LowerMinifloatCast>;

/// Lowers every cast from or to an fp8 format the backend does not convert natively onto the
/// software polyfill, through `f32`.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerMinifloatCast {
    native: EnumSet<Fp8Format>,
    container: Fp8Container,
}

impl LowerMinifloatCast {
    fn emulated(&self, ctx: &Context, value: impl Typed) -> Option<Fp8Format> {
        Fp8Format::of_type(ctx, value.scalar_ty(ctx))
            .filter(|format| !self.native.contains(*format))
    }
}

impl MatchRewrite for LowerMinifloatCast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<CastOp>(ctx) {
            return false;
        }
        self.emulated(ctx, op.operand(ctx, 0)).is_some()
            || self.emulated(ctx, op.result(ctx)).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let input = op.operand(ctx, 0);
        let result_ty = op.result(ctx).get_type(ctx);
        let lanes = input.vector_size(ctx);
        debug_assert_eq!(
            lanes,
            result_ty.vector_size(ctx),
            "A cast keeps its vectorization, so one `N` describes both sides"
        );
        scope.register_size::<N>(lanes);

        // Bool sources and targets go through the `f32` cast the backends already lower.
        let mut value = input;
        if let Some(format) = self.emulated(ctx, input) {
            value = self.decode(&scope, value, format);
        }
        let value = match self.emulated(ctx, result_ty) {
            Some(format) => self.encode(&scope, value, format, result_ty),
            None => cast_value(&scope, value, result_ty),
        };
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

impl LowerMinifloatCast {
    fn decode(&self, scope: &Scope, value: Value, format: Fp8Format) -> Value {
        let bits = match self.container {
            Fp8Container::Bytes => {
                let bytes =
                    reinterpret_value(scope, value, Vector::<u8, N>::__expand_as_type(scope));
                cast_value(scope, bytes, Vector::<u32, N>::__expand_as_type(scope))
            }
            Fp8Container::Words => {
                let words = reinterpret_value(scope, value, words_type(scope));
                unpack_words::expand::<N, W>(scope, words.into()).read_value(scope)
            }
        };
        fp8_bits_to_f32::expand::<N>(scope, bits.into(), format).read_value(scope)
    }

    fn encode(
        &self,
        scope: &Scope,
        value: Value,
        format: Fp8Format,
        result_ty: TypeHandle,
    ) -> Value {
        let value = cast_value(scope, value, Vector::<f32, N>::__expand_as_type(scope));
        let bits = f32_to_fp8_bits::expand::<N>(scope, value.into(), format).read_value(scope);
        let container = match self.container {
            Fp8Container::Bytes => {
                cast_value(scope, bits, Vector::<u8, N>::__expand_as_type(scope))
            }
            Fp8Container::Words => {
                register_words_size(scope);
                pack_words::expand::<N, W>(scope, bits.into()).read_value(scope)
            }
        };
        reinterpret_value(scope, container, result_ty)
    }
}

pub type LowerMinifloatComparePass = MatchRewritePass<LowerMinifloatCompare>;

/// Lowers fp8 equality onto the lanes' bit patterns.
///
/// No backend compares fp8 as a float. `VK_EXT_shader_float8` allows conversion, cooperative
/// matrix multiply, and the operations that only move bits around, so a float comparison is out
/// even where fp8 is native; without it fp8 is an integer, which a float comparison cannot read
/// either. Comparing the bits is what a CUDA kernel already gets, where fp8 is the raw
/// `__nv_fp8_storage_t` byte and `__nv_fp8_e4m3` declares no comparison operators at all.
///
/// Bit equality parts from float equality in exactly two places: `0.0` and `-0.0` are equal as
/// floats and different as bits, and a NaN equals itself here where a float NaN does not. Scale
/// factors, where fp8 sees most of its use, are non-negative and never NaN, so neither case
/// reaches them.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerMinifloatCompare {
    container: Fp8Container,
}

impl MatchRewrite for LowerMinifloatCompare {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        if !op.is_op::<FEqualOp>(ctx) && !op.is_op::<FNotEqualOp>(ctx) {
            return false;
        }
        Fp8Format::of_type(ctx, op.operand(ctx, 0).scalar_ty(ctx)).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let equal = op.is_op::<FEqualOp>(ctx);
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lhs = op.operand(scope.ctx(), 0);
        let rhs = op.operand(scope.ctx(), 1);
        scope.register_size::<N>(lhs.vector_size(scope.ctx()));

        let lhs = self.lanes(&scope, lhs);
        let rhs = self.lanes(&scope, rhs);
        let value = match self.container {
            Fp8Container::Bytes => compare_lanes::<u8>(&scope, equal, lhs, rhs),
            Fp8Container::Words => compare_lanes::<u32>(&scope, equal, lhs, rhs),
        };
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

impl LowerMinifloatCompare {
    /// One lane per lane, in whichever integer the container leaves them addressable in. Packed
    /// lanes have to come apart first: comparing the words would answer once for four lanes.
    fn lanes(&self, scope: &Scope, value: Value) -> Value {
        match self.container {
            Fp8Container::Bytes => {
                reinterpret_value(scope, value, Vector::<u8, N>::__expand_as_type(scope))
            }
            Fp8Container::Words => {
                let words = reinterpret_value(scope, value, words_type(scope));
                unpack_words::expand::<N, W>(scope, words.into()).read_value(scope)
            }
        }
    }
}

fn compare_lanes<T: Int>(scope: &Scope, equal: bool, lhs: Value, rhs: Value) -> Value {
    match equal {
        true => bits_equal::expand::<T>(scope, lhs.into(), rhs.into()).read_value(scope),
        false => bits_not_equal::expand::<T>(scope, lhs.into(), rhs.into()).read_value(scope),
    }
}

#[cube]
fn bits_equal<T: Int>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<bool, N> {
    lhs.equal(&rhs)
}

#[cube]
fn bits_not_equal<T: Int>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<bool, N> {
    lhs.not_equal(&rhs)
}

define_size!(W);

const LANES_PER_WORD: usize = (u32::BITS / u8::BITS) as usize;

/// Registers `W`, the word count of an `N`-lane fp8 vector, so that `Vector<u32, W>` names the
/// words those lanes are packed into.
fn register_words_size(scope: &Scope) {
    let lanes = N::__expand_value(scope);
    assert!(
        lanes.is_multiple_of(LANES_PER_WORD),
        "fp8 is packed four lanes to a u32 on this backend: vectors of {lanes} lanes are not \
         supported, use a vector size that is a multiple of {LANES_PER_WORD}"
    );
    scope.register_size::<W>(lanes / LANES_PER_WORD);
}

/// [`register_words_size`], then the word type itself.
fn words_type(scope: &Scope) -> TypeHandle {
    register_words_size(scope);
    Vector::<u32, W>::__expand_as_type(scope)
}

#[cube]
fn unpack_words<N: Size, W: Size>(words: Vector<u32, W>) -> Vector<u32, N> {
    let mut lanes = Vector::<u32, N>::empty();
    #[unroll]
    for lane in 0..N::value() {
        let word = words.extract(lane / LANES_PER_WORD);
        let shift = comptime![(lane % LANES_PER_WORD) as u32 * u8::BITS];
        lanes.insert(lane, (word >> shift) & 0xFF);
    }
    lanes
}

#[cube]
fn pack_words<N: Size, W: Size>(lanes: Vector<u32, N>) -> Vector<u32, W> {
    let mut words = Vector::<u32, W>::empty();
    #[unroll]
    for index in 0..W::value() {
        let mut word = 0u32;
        #[unroll]
        for offset in 0..LANES_PER_WORD {
            let shift = comptime![offset as u32 * u8::BITS];
            word |= (lanes.extract(index * LANES_PER_WORD + offset) & 0xFF) << shift;
        }
        words.insert(index, word);
    }
    words
}
