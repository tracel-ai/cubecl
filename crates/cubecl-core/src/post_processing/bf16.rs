//! Software `bf16` for backends that carry it as a 16-bit integer.
//!
//! Such a backend stores and moves `bf16` lanes as `u16` bit patterns, which every operation that
//! only moves bits (loads, stores, selects, shuffles, vector construction) handles unchanged. What
//! remains is arithmetic: [`PromoteBf16`] rebuilds each float operation on `bf16` at `f32` and
//! rounds its result back, so every operation rounds once as a native `bf16` instruction would.
//! [`LowerBf16Cast`] then lowers every conversion from or to `bf16` onto the bit patterns.

use alloc::{string::ToString, vec, vec::Vec};
use cubecl_ir::{
    NamedRewrite, Scope,
    dialect::{
        base::OperationPtrExt,
        general::{CastOp, PrintfOp},
        plane, vector,
    },
    interfaces::{MaterializableOp, TypeExt, TypedExt},
    prelude::*,
    try_cast_op,
    types::scalar::{BFloat16Type, Float32Type},
};
use pliron::r#type::TypeHandle;

use crate::{self as cubecl, prelude::*};

define_size!(N);

const BF16_SHIFT: u32 = u32::BITS - u16::BITS;
const F32_MAGNITUDE_MASK: u32 = u32::MAX >> 1;
const F32_INFINITY_BITS: u32 = f32::INFINITY.to_bits();
/// The top mantissa bit of a `bf16`, which makes a NaN quiet.
const BF16_QUIET_BIT: u32 = 1 << 6;

/// Bits above the low half-word are ignored.
#[cube]
pub fn bf16_bits_to_f32<N: Size>(bits: Vector<u32, N>) -> Vector<f32, N> {
    // `bf16` is the top half of an `f32`, so widening is exact.
    Vector::<f32, N>::reinterpret(bits << Vector::new(BF16_SHIFT))
}

/// Round to nearest even, and a NaN stays a NaN with its sign, as `half::bf16::from_f32` does.
#[cube]
pub fn f32_to_bf16_bits<N: Size>(value: Vector<f32, N>) -> Vector<u32, N> {
    let bits = Vector::<u32, N>::reinterpret(value);
    let truncated = bits >> Vector::new(BF16_SHIFT);
    // A carry out of the mantissa lands in the exponent, which is the correct rounding up to the
    // next binade, or to infinity past the largest finite value.
    let lsb = truncated & Vector::new(1u32);
    let rounded = (bits + Vector::new(0x7FFFu32) + lsb) >> Vector::new(BF16_SHIFT);
    // Rounding a NaN whose payload sits in the dropped bits would land on infinity instead.
    let is_nan =
        (bits & Vector::new(F32_MAGNITUDE_MASK)).greater_than(&Vector::new(F32_INFINITY_BITS));
    select_many(is_nan, truncated | Vector::new(BF16_QUIET_BIT), rounded)
}

pub type PromoteBf16Pass = MatchRewritePass<PromoteBf16>;

/// Rebuilds float operations on `bf16` at `f32`, narrowing their `bf16` results back.
///
/// The operations are listed rather than inferred: an operation missing here keeps its `bf16`
/// operands and fails to lower as float arithmetic on an integer, where promoting an operation
/// that only moves bits would silently change what it moves.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct PromoteBf16;

impl MatchRewrite for PromoteBf16 {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        let touches_bf16 = op
            .operands(ctx)
            .into_iter()
            .any(|value| is_bf16(ctx, value))
            || op.deref(ctx).results().any(|value| is_bf16(ctx, value));
        touches_bf16 && (is_float_op(ctx, op) || op.is_op::<PrintfOp>(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let operands = widen_operands(&scope, op);
        // A `printf` has no result to narrow and is not rebuilt.
        if op.is_op::<PrintfOp>(scope.ctx()) {
            print_widened(&scope, op, operands);
            return Ok(());
        }
        let narrowed = compute_at_f32(&scope, op, operands);
        rewriter.replace_operation_with_values(ctx, op, narrowed);
        Ok(())
    }
}

pub type LowerBf16CastPass = MatchRewritePass<LowerBf16Cast>;

/// Lowers every cast from or to `bf16` onto its `u16` bit pattern, through `f32`.
#[derive(new, Clone, Copy, Debug, Default, NamedRewrite)]
pub struct LowerBf16Cast;

impl MatchRewrite for LowerBf16Cast {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<CastOp>(ctx)
            && (is_bf16(ctx, op.operand(ctx, 0)) || is_bf16(ctx, op.result(ctx)))
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

        // Bool and integer sources and targets go through the `f32` cast the backend lowers.
        let mut value = input;
        if is_bf16(scope.ctx(), input) {
            value = decode(&scope, value);
        }
        let value = match is_bf16(scope.ctx(), result_ty) {
            true => encode(&scope, value, result_ty),
            false => cast_value(&scope, value, result_ty),
        };
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

fn is_bf16(ctx: &Context, value: impl Typed) -> bool {
    value
        .try_get_scalar_ty(ctx)
        .is_some_and(|scalar| scalar.deref(ctx).is::<BFloat16Type>())
}

fn is_float_op(ctx: &Context, op: Ptr<Operation>) -> bool {
    let opid = Operation::get_opid(op, ctx);
    // Every operation of these two dialects that accepts a float computes on it.
    let dialect = opid.dialect.to_string();
    if dialect == "math" || dialect == "cmp" {
        return true;
    }
    op.is_op::<vector::MagnitudeOp>(ctx)
        || op.is_op::<vector::NormalizeOp>(ctx)
        || op.is_op::<vector::FSumOp>(ctx)
        || op.is_op::<vector::FDotOp>(ctx)
        || op.is_op::<plane::FSumOp>(ctx)
        || op.is_op::<plane::InclusiveFSumOp>(ctx)
        || op.is_op::<plane::ExclusiveFSumOp>(ctx)
        || op.is_op::<plane::FProdOp>(ctx)
        || op.is_op::<plane::InclusiveFProdOp>(ctx)
        || op.is_op::<plane::ExclusiveFProdOp>(ctx)
        || op.is_op::<plane::FMinOp>(ctx)
        || op.is_op::<plane::FMaxOp>(ctx)
}

/// The operands of `op`, each `bf16` one cast to `f32`.
fn widen_operands(scope: &Scope, op: Ptr<Operation>) -> Vec<Value> {
    op.operands(scope.ctx())
        .into_iter()
        .map(|value| match is_bf16(scope.ctx(), value) {
            true => cast_value(scope, value, f32_shaped(scope, value.get_type(scope.ctx()))),
            false => value,
        })
        .collect()
}

/// Points `op` at its widened operands in place: `f32` prints every `bf16` exactly.
fn print_widened(scope: &Scope, op: Ptr<Operation>, operands: Vec<Value>) {
    for (index, wide) in operands.into_iter().enumerate() {
        if op.operand(scope.ctx(), index) != wide {
            Operation::replace_operand(op, scope.ctx(), index, wide);
        }
    }
}

/// Rebuilds `op` on its widened operands with `f32` in place of each `bf16` result, and returns
/// the results narrowed back to the types `op` declared.
fn compute_at_f32(scope: &Scope, op: Ptr<Operation>, operands: Vec<Value>) -> Vec<Value> {
    let results: Vec<Value> = op.deref(scope.ctx()).results().collect();
    let attributes = op.deref(scope.ctx()).attributes.clone();
    let result_tys = results
        .iter()
        .map(|value| {
            let ty = value.get_type(scope.ctx());
            match is_bf16(scope.ctx(), *value) {
                true => f32_shaped(scope, ty),
                false => ty,
            }
        })
        .collect();

    let dyn_op = op.dyn_op(scope.ctx());
    let rematerialize = try_cast_op!(dyn_op, scope.ctx(), dyn MaterializableOp);
    let wide_op = rematerialize.materialize(scope.ctx_mut(), result_tys, operands, attributes);
    scope.inserter().append_operation(scope.ctx(), wide_op);

    results
        .iter()
        .enumerate()
        .map(|(index, narrow)| {
            let wide = wide_op.deref(scope.ctx()).get_result(index);
            cast_value(scope, wide, narrow.get_type(scope.ctx()))
        })
        .collect()
}

fn f32_shaped(scope: &Scope, ty: TypeHandle) -> TypeHandle {
    let f32_ty = Float32Type::get(scope.ctx()).to_handle();
    ty.with_scalar(scope.ctx_mut(), f32_ty)
}

/// `N` lanes of `bf16` widened to `f32`.
fn decode(scope: &Scope, value: Value) -> Value {
    let halves = reinterpret_value(scope, value, Vector::<u16, N>::__expand_as_type(scope));
    let bits = cast_value(scope, halves, Vector::<u32, N>::__expand_as_type(scope));
    bf16_bits_to_f32::expand::<N>(scope, bits.into()).read_value(scope)
}

/// `N` lanes of any type the backend casts to `f32`, narrowed to `result_ty`'s `bf16`.
fn encode(scope: &Scope, value: Value, result_ty: TypeHandle) -> Value {
    let value = cast_value(scope, value, Vector::<f32, N>::__expand_as_type(scope));
    let bits = f32_to_bf16_bits::expand::<N>(scope, value.into()).read_value(scope);
    let halves = cast_value(scope, bits, Vector::<u16, N>::__expand_as_type(scope));
    reinterpret_value(scope, halves, result_ty)
}
