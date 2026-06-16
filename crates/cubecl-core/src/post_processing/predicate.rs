use alloc::vec;
use core::{f32, f64};

use crate as cubecl;
use cubecl_ir::{
    ElemType, FloatKind, Scope, UIntKind,
    dialect::{
        base::OperationPtrExt,
        math::{IsInfOp, IsNanOp},
    },
    interfaces::TypedExt,
    pliron::{
        irbuild::{
            dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
            rewriter::Rewriter,
        },
        op::{op_cast, op_impls},
        op_interface, op_interface_impl,
        operation::Operation,
        prelude::{Context, Ptr, Result},
        value::Value,
    },
    types::scalar::FloatType,
    verify_op_succ,
};
use half::{bf16, f16};

use crate::prelude::*;

define_scalar!(ElemA);
define_scalar!(IntB);
define_size!(SizeA);

pub struct ReplacePredicates;

impl DialectConversion for ReplacePredicates {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn PredicateOp>(&*op.dyn_op(ctx))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let input = op.operand(ctx, 0);
        let dyn_op = op.dyn_op(ctx);
        let predicate = op_cast::<dyn PredicateOp>(&*dyn_op).unwrap();
        let new_value = predicate.run_polyfill(&scope, input);
        rewriter.replace_operation_with_values(ctx, op, vec![new_value]);
        Ok(())
    }
}

#[op_interface]
trait PredicateOp {
    verify_op_succ!();
    fn run_polyfill(&self, scope: &Scope, input: Value) -> Value;
}

#[op_interface_impl]
impl PredicateOp for IsNanOp {
    fn run_polyfill(&self, scope: &Scope, input: Value) -> Value {
        run_polyfill(scope, input, is_nan::expand::<ElemA, IntB, SizeA>)
    }
}

#[op_interface_impl]
impl PredicateOp for IsInfOp {
    fn run_polyfill(&self, scope: &Scope, input: Value) -> Value {
        run_polyfill(scope, input, is_inf::expand::<ElemA, IntB, SizeA>)
    }
}

fn run_polyfill<T: CubePrimitive, O: CubePrimitive>(
    scope: &Scope,
    input: Value,
    mut polyfill: impl FnMut(&Scope, NativeExpand<T>, u32, u32) -> NativeExpand<O>,
) -> Value {
    scope.register_value_type::<ElemA, SizeA>(input);

    let ty = input.scalar_ty(scope.ctx()).deref(scope.ctx());
    let ty = *ty.downcast_ref::<FloatType>().expect("Should be float");

    let (unsigned_ty, bit_width, mantissa_bits) = match ty.encoding {
        FloatKind::F64 => (UIntKind::U64, f64::size_bits(), f64::MANTISSA_DIGITS - 1),
        FloatKind::F32 => (UIntKind::U32, f32::size_bits(), f32::MANTISSA_DIGITS - 1),
        FloatKind::F16 => (UIntKind::U16, f16::size_bits(), f16::MANTISSA_DIGITS - 1),
        FloatKind::BF16 => (UIntKind::U16, bf16::size_bits(), bf16::MANTISSA_DIGITS - 1),
        _ => unreachable!(),
    };
    scope.register_type::<IntB>(ElemType::UInt(unsigned_ty).into());

    let exp_bits = bit_width as u32 - mantissa_bits - 1;

    polyfill(scope, input.into(), mantissa_bits, exp_bits).value(scope)
}

#[cube]
fn is_nan<F: Float, U: Int, N: Size>(
    x: Vector<F, N>,
    #[comptime] mantissa_bits: u32,
    #[comptime] exp_bits: u32,
) -> Vector<bool, N> {
    // Need to mark as u64 otherwise it is coerced into i32 which does not fit the values for f64
    let inf_bits = comptime![((1u64 << exp_bits as u64) - 1u64) << mantissa_bits as u64];
    let abs_mask = comptime![(1u64 << (exp_bits as u64 + mantissa_bits as u64)) - 1u64];

    let bits: Vector<U, N> = Vector::<U, N>::reinterpret(x);

    let abs_bits = bits & Vector::new(U::cast_from(abs_mask));
    let inf_bits = Vector::new(U::cast_from(inf_bits));

    abs_bits.greater_than(&inf_bits)
}

// Same trick as NaN detection following IEEE 754, but check for all 0 bits equality
#[cube]
fn is_inf<F: Float, U: Int, N: Size>(
    x: Vector<F, N>,
    #[comptime] mantissa_bits: u32,
    #[comptime] exp_bits: u32,
) -> Vector<bool, N> {
    // Need to mark as u64 otherwise it is coerced into i32 which does not fit the values for f64
    let inf_bits = comptime![((1u64 << exp_bits as u64) - 1u64) << mantissa_bits as u64];
    let abs_mask = comptime![(1u64 << (exp_bits as u64 + mantissa_bits as u64)) - 1u64];

    let bits: Vector<U, N> = Vector::<U, N>::reinterpret(x);

    let abs_bits = bits & Vector::new(U::cast_from(abs_mask));
    let inf_bits = Vector::new(U::cast_from(inf_bits));

    abs_bits.equal(&inf_bits)
}
