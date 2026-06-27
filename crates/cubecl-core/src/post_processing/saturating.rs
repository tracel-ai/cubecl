use crate as cubecl;
use alloc::vec;
use cubecl_ir::{
    ElemType, Scope, UIntKind,
    dialect::{
        base::OperationPtrExt,
        math::{SaturatingAddOp, SaturatingSubOp},
    },
    interfaces::TypedExt,
    prelude::*,
};

use crate::prelude::*;

define_scalar!(ElemA);
define_scalar!(ElemB);
define_size!(SizeA);

pub type LowerSaturatingArithmeticPass = MatchRewritePass<LowerSaturatingArithmetic>;

/// Replaces saturating arithmetic with a performant polyfill
#[derive(new, Debug, Default)]
pub struct LowerSaturatingArithmetic;

impl MatchRewrite for LowerSaturatingArithmetic {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.is_op::<SaturatingAddOp>(ctx) || op.is_op::<SaturatingSubOp>(ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let lhs = op.deref(ctx).get_operand(0);
        let rhs = op.deref(ctx).get_operand(1);

        let value = if op.is_op::<SaturatingAddOp>(ctx) {
            if lhs.scalar_ty(ctx).is_int(ctx) {
                run_polyfill(
                    (&scope, lhs, rhs),
                    saturating_add_signed::expand::<ElemA, ElemB, SizeA>,
                )
            } else if lhs.scalar_ty(ctx).is_uint(ctx) {
                run_polyfill(
                    (&scope, lhs, rhs),
                    saturating_add_unsigned::expand::<ElemA, SizeA>,
                )
            } else {
                unreachable!("Should be int or uint")
            }
        } else if op.is_op::<SaturatingSubOp>(ctx) {
            if lhs.scalar_ty(ctx).is_int(ctx) {
                run_polyfill(
                    (&scope, lhs, rhs),
                    saturating_sub_signed::expand::<ElemA, ElemB, SizeA>,
                )
            } else if lhs.scalar_ty(ctx).is_uint(ctx) {
                run_polyfill(
                    (&scope, lhs, rhs),
                    saturating_sub_unsigned::expand::<ElemA, SizeA>,
                )
            } else {
                unreachable!("Should be int or uint")
            }
        } else {
            unreachable!()
        };
        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

fn run_polyfill<T: CubePrimitive>(
    (scope, lhs, rhs): (&Scope, Value, Value),
    mut polyfill: impl FnMut(&Scope, NativeExpand<T>, NativeExpand<T>) -> NativeExpand<T>,
) -> Value {
    scope.register_value_type::<ElemA, SizeA>(lhs);
    if lhs.scalar_ty(scope.ctx()).is_int(scope.ctx()) {
        let unsigned_ty = match lhs.scalar_ty(scope.ctx()).size(scope.ctx()) {
            1 => UIntKind::U8,
            2 => UIntKind::U16,
            4 => UIntKind::U32,
            8 => UIntKind::U64,
            _ => unreachable!("Unsupported width"),
        };
        scope.register_type::<ElemB>(ElemType::UInt(unsigned_ty))
    }

    polyfill(scope, lhs.into(), rhs.into()).value(scope)
}

#[cube]
fn saturating_add_unsigned<U: Int, N: Size>(a: Vector<U, N>, b: Vector<U, N>) -> Vector<U, N> {
    let c = a.min(!b);
    c + b
}

#[cube]
fn saturating_sub_unsigned<U: Int, N: Size>(a: Vector<U, N>, b: Vector<U, N>) -> Vector<U, N> {
    let a = a.max(b);
    a - b
}

/// Don't ask me how this works
/// <https://locklessinc.com/articles/sat_arithmetic/>
#[cube]
fn saturating_add_signed<I: Int, U: Int, N: Size>(
    x: Vector<I, N>,
    y: Vector<I, N>,
) -> Vector<I, N> {
    let bit_width = I::size_bits();
    let shift = Vector::<U, N>::new(U::new(comptime![(bit_width - 1) as i64]));

    let ux = Vector::<U, N>::cast_from(x);
    let uy = Vector::<U, N>::cast_from(y);
    let res = ux + uy;
    let ux = (ux >> shift) + Vector::<U, N>::cast_from(I::max_value());
    let zero = Vector::new(I::new(0));
    let cond = Vector::<I, N>::cast_from((ux ^ uy) | !(uy ^ res)).greater_equal(&zero);
    select_many(cond, Vector::cast_from(ux), Vector::cast_from(res))
}

/// Don't ask me how this works
/// <https://locklessinc.com/articles/sat_arithmetic/>
#[cube]
fn saturating_sub_signed<I: Int, U: Int, N: Size>(
    x: Vector<I, N>,
    y: Vector<I, N>,
) -> Vector<I, N> {
    let bit_width = I::size_bits();
    let shift = Vector::<U, N>::new(U::new(comptime![(bit_width - 1) as i64]));

    let ux = Vector::<U, N>::cast_from(x);
    let uy = Vector::<U, N>::cast_from(y);
    let res = ux - uy;
    let ux = (ux >> shift) + Vector::<U, N>::cast_from(I::max_value());
    let zero = Vector::new(I::new(0));
    let cond = Vector::<I, N>::cast_from((ux ^ uy) & (ux ^ res)).less_than(&zero);
    select_many(cond, Vector::cast_from(ux), Vector::cast_from(res))
}
