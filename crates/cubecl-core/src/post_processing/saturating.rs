use crate as cubecl;
use alloc::vec;
use cubecl_ir::{
    Scope,
    dialect::{
        base::OperationPtrExt,
        math::{SaturatingSAddOp, SaturatingSSubOp, SaturatingUAddOp, SaturatingUSubOp},
    },
    interfaces::TypedExt,
    prelude::*,
};
use pliron::builtin::types::{IntegerType, Signedness};

use crate::prelude::*;

define_scalar!(Elem);
define_scalar!(ElemU);
define_size!(N);

pub type LowerSaturatingArithmeticPass = MatchRewritePass<LowerSaturatingArithmetic>;

#[op_interface]
trait SaturatingOp {
    verify_op_succ!();
    fn run_polyfill(&self, args: (&Scope, Value, Value)) -> Value;
}

#[op_interface_impl]
impl SaturatingOp for SaturatingSAddOp {
    fn run_polyfill(&self, args: (&Scope, Value, Value)) -> Value {
        run_polyfill(args, saturating_add_signed::expand::<Elem, ElemU, N>)
    }
}

#[op_interface_impl]
impl SaturatingOp for SaturatingUAddOp {
    fn run_polyfill(&self, args: (&Scope, Value, Value)) -> Value {
        run_polyfill(args, saturating_add_unsigned::expand::<Elem, N>)
    }
}

#[op_interface_impl]
impl SaturatingOp for SaturatingSSubOp {
    fn run_polyfill(&self, args: (&Scope, Value, Value)) -> Value {
        run_polyfill(args, saturating_sub_signed::expand::<Elem, ElemU, N>)
    }
}

#[op_interface_impl]
impl SaturatingOp for SaturatingUSubOp {
    fn run_polyfill(&self, args: (&Scope, Value, Value)) -> Value {
        run_polyfill(args, saturating_sub_unsigned::expand::<Elem, N>)
    }
}

/// Replaces saturating arithmetic with a performant polyfill
#[derive(new, Debug, Default)]
pub struct LowerSaturatingArithmetic;

impl MatchRewrite for LowerSaturatingArithmetic {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op.impls::<dyn SaturatingOp>(ctx)
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

        let dyn_op = op.dyn_op(ctx);
        let sat_op = op_cast::<dyn SaturatingOp>(&*dyn_op).unwrap();
        let value = sat_op.run_polyfill((&scope, lhs, rhs));

        rewriter.replace_operation_with_values(ctx, op, vec![value]);
        Ok(())
    }
}

fn run_polyfill<T: CubePrimitive>(
    (scope, lhs, rhs): (&Scope, Value, Value),
    mut polyfill: impl FnMut(&Scope, NativeExpand<T>, NativeExpand<T>) -> NativeExpand<T>,
) -> Value {
    let ctx = scope.ctx();
    let width = lhs.scalar_ty(ctx).size_bits(ctx);
    let unsigned_ty = IntegerType::get(ctx, width as u32, Signedness::Unsigned);
    scope.register_value_type::<Elem, N>(lhs);
    scope.register_value_type::<ElemU, ()>(unsigned_ty.to_handle());

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
