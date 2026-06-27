use core::any::type_name;

use cubecl_ir::{
    dialect::{
        bitwise::{
            CountOnesOp, FindFirstSetOp, LeadingZerosBitsOp, ReverseBitsOp, TrailingZerosBitsOp,
        },
        general::CastOp,
    },
    interfaces::TypedExt,
    prelude::*,
    types::{VectorType, scalar::UIntType},
};
use pliron::irbuild::match_rewrite::{RewriterOrder, apply_match_rewrite};

use crate::{self as cubecl, prelude::*};

define_scalar!(T);
define_size!(N);

pub struct PromoteBitwisePass;

impl Pass for PromoteBitwisePass {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        res.ir_changed |= apply(ctx, PromoteCountOnesAndFfs, op)?;
        res.ir_changed |= apply(ctx, PromoteLeadingZerosBitsOp, op)?;
        res.ir_changed |= apply(ctx, PromoteTrailingZerosBitsOp, op)?;
        res.ir_changed |= apply(ctx, PromoteReverseBitsOp, op)?;

        Ok(res)
    }
}

fn apply<M: MatchRewrite>(
    ctx: &mut Context,
    mut match_rewrite: M,
    op: Ptr<Operation>,
) -> Result<IRStatus> {
    apply_match_rewrite(ctx, &mut match_rewrite, RewriterOrder::default(), op)
}

// No special handling beyond zero extend
struct PromoteCountOnesAndFfs;
impl MatchRewrite for PromoteCountOnesAndFfs {
    fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
        (op.is_op::<CountOnesOp>(ctx) || op.is_op::<FindFirstSetOp>(ctx))
            && lhs_is_small_int(ctx, op)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut MatchRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let promoted = zero_extend(&scope, op.operand(ctx, 0));
        op.operand(ctx, 0)
            .replace_use_with(ctx, op.operand_as_use(ctx, 0), &promoted);
        Ok(())
    }
}

macro_rules! with_cube_polyfill {
    ($op: ty, $polyfill: ident) => {paste::paste!{
        struct [<Promote $op>];
        impl MatchRewrite for [<Promote $op>] {
            fn r#match(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
                op.is_op::<$op>(ctx) && lhs_is_small_int(ctx, op)
            }

            fn rewrite(
                &mut self,
                ctx: &mut Context,
                rewriter: &mut MatchRewriter,
                op: Ptr<Operation>,
            ) -> Result<()> {
                let scope = Scope::from_context_and_inserter(ctx, rewriter);
                scope.register_value_type::<T, N>(op.operand(ctx, 0));
                let input = zero_extend(&scope, op.operand(ctx, 0));
                let new_res = $polyfill::expand::<T, N>(&scope, input.into()).read_value(&scope);
                rewriter.replace_operation_with_values(ctx, op, vec![new_res]);
                Ok(())
            }
        }
    }};
}

#[cube]
fn leading_zeros<T: Int, N: Size>(value: Vector<u32, N>) -> Vector<u32, N> {
    let adjust_bits = 32 - T::size_bits().comptime() as u32;
    value.leading_zeros() - Vector::new(adjust_bits)
}

with_cube_polyfill!(LeadingZerosBitsOp, leading_zeros);

#[cube]
fn trailing_zeros<T: Int, N: Size>(value: Vector<u32, N>) -> Vector<u32, N> {
    let size = Vector::new(T::size_bits().comptime() as u32);
    select_many(value.equal(&Vector::new(0)), size, value.trailing_zeros())
}

with_cube_polyfill!(TrailingZerosBitsOp, trailing_zeros);

#[cube]
fn reverse_bits<T: Int, N: Size>(value: Vector<u32, N>) -> Vector<T, N> {
    let shift = Vector::new(32 - T::size_bits().comptime() as u32);
    Vector::cast_from(value.reverse_bits() >> shift)
}

with_cube_polyfill!(ReverseBitsOp, reverse_bits);

fn lhs_is_small_int(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.operand(ctx, 0).scalar_ty(ctx).size(ctx) < size_of::<u32>()
}

fn zero_extend(scope: &Scope, value: Value) -> Value {
    let ctx = scope.ctx_mut();
    let scalar = value.scalar_ty(ctx);
    let unsigned = match value.scalar_ty(ctx).is_int(ctx) {
        true => {
            let scalar_ty = UIntType::get(ctx, scalar.size_bits(ctx)).to_handle();
            let ty = with_scalar_ty(ctx, value.get_type(ctx), scalar_ty);
            let cast = CastOp::new(ctx, ty, value);
            scope.register(&cast);
            cast.get_result(ctx)
        }
        false => value,
    };
    let scalar_ty = UIntType::get(ctx, 32).to_handle();
    let ty = with_scalar_ty(ctx, value.get_type(ctx), scalar_ty);
    let cast = CastOp::new(ctx, ty, unsigned);
    scope.register(&cast);
    cast.get_result(ctx)
}

fn with_scalar_ty(ctx: &Context, ty: TypeHandle, scalar: TypeHandle) -> TypeHandle {
    if let Some(vector) = ty.deref(ctx).downcast_ref::<VectorType>() {
        VectorType::get(ctx, scalar, vector.vectorization).to_handle()
    } else {
        scalar
    }
}
