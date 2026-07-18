use cubecl_core::{
    self as cubecl, define_scalar, define_size,
    num_traits::{One, Zero},
    prelude::*,
};
use cubecl_ir::{Scope, dialect::general::CastOp, interfaces::TypedExt, prelude::*};
use pliron::printable::Printable;
use pliron_spirv::{ops::*, types::FloatType};

use crate::{lower::LowerOp, ops::to_spirv_dialect::ToSpirvDialectOp, types::ty_to_spirv_dialect};

define_scalar!(T);
define_size!(N);

#[op_interface_impl]
impl ToSpirvDialectOp for CastOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let from_ty = operands_info
            .lookup_most_recent_type(input)
            .unwrap_or_else(|| input.get_type(ctx));
        let value = cast(ctx, rewriter, input, from_ty, self.result_type(ctx));
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![value]);
        Ok(())
    }
}

pub(crate) fn cast(
    ctx: &mut Context,
    rewriter: &mut impl Rewriter,
    from: Value,
    from_ty: TypeHandle,
    to: TypeHandle,
) -> Value {
    let in_ty = from_ty.element_ty(ctx).scalar_ty(ctx);
    let to_ty = to.element_ty(ctx).scalar_ty(ctx);
    let in_ty_spirv = ty_to_spirv_dialect(ctx, in_ty);
    let to_ty_spirv = ty_to_spirv_dialect(ctx, to_ty);
    let out_ty = ty_to_spirv_dialect(ctx, to);

    let in_float = in_ty_spirv.deref(ctx).is::<FloatType>();
    let in_sint = in_ty.is_signed_int(ctx);
    let in_uint = in_ty.is_unsigned_int(ctx) || in_ty.is_index(ctx);
    let out_float = to_ty_spirv.deref(ctx).is::<FloatType>();
    let out_sint = to_ty.is_signed_int(ctx);
    let out_uint = to_ty.is_unsigned_int(ctx) || to_ty.is_index(ctx);

    if in_ty_spirv == to_ty_spirv {
        from
    } else if in_float && out_uint {
        let conv = ConvertFToUOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else if in_float && out_sint {
        let conv = ConvertFToSOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else if in_sint && out_float {
        let conv = ConvertSToFOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else if in_uint && out_float {
        let conv = ConvertUToFOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else if in_uint && (to_ty.is_int(ctx) || to_ty.is_index(ctx)) {
        let conv = UConvertOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else if in_sint && (to_ty.is_int(ctx) || to_ty.is_index(ctx)) {
        let conv = SConvertOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else if in_float && out_float {
        let conv = FConvertOp::new(ctx, out_ty, from);
        rewriter.append_op_with_result(ctx, &conv)
    } else {
        panic!(
            "cast from {} to {} not supported",
            from.get_type(ctx).disp(ctx),
            to.disp(ctx)
        )
    }
}

#[op_interface_impl]
impl LowerOp for CastOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        self.input(ctx).scalar_ty(ctx).is_bool(ctx)
            || self.result_type(ctx).scalar_ty(ctx).is_bool(ctx)
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let input = self.input(scope.ctx());
        let result_ty = self.result_type(scope.ctx());
        let value = if input.scalar_ty(scope.ctx()).is_bool(scope.ctx()) {
            scope.register_value_type::<T, N>(result_ty);
            bool_to_numeric::expand::<T>(scope, input.into()).read_value(scope)
        } else {
            scope.register_value_type::<T, N>(input);
            numeric_to_bool::expand::<T>(scope, input.into()).read_value(scope)
        };
        vec![value]
    }
}

#[cube]
fn bool_to_numeric<T: Numeric>(input: Vector<bool, N>) -> Vector<T, N> {
    select_many(input, Vector::one(), Vector::zero())
}

#[cube]
fn numeric_to_bool<T: Numeric>(input: Vector<T, N>) -> Vector<bool, N> {
    input.not_equal(&Vector::zero())
}
