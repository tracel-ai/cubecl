use cubecl_core::{
    self as cubecl, define_scalar, define_size,
    ir::types::Fp8Format,
    num_traits::{One, Zero},
    prelude::*,
};
use cubecl_ir::{Scope, dialect::general::CastOp, interfaces::TypedExt, prelude::*};
use pliron::printable::Printable;
use pliron_spirv::{decorations::DecoratableOp, ops::*, types::FloatType};

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
    // fp8 without a SPIR-V float type is emulated: the value is its byte, zero extended.
    let in_uint = in_ty.is_unsigned_int(ctx)
        || in_ty.is_index(ctx)
        || (!in_float && Fp8Format::of_type(ctx, in_ty).is_some());
    let out_float = to_ty_spirv.deref(ctx).is::<FloatType>();
    let out_sint = to_ty.is_signed_int(ctx);
    let out_uint = to_ty.is_unsigned_int(ctx)
        || to_ty.is_index(ctx)
        || (!out_float && Fp8Format::of_type(ctx, to_ty).is_some());

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
        // Native fp8 otherwise sends overflow to NaN (e4m3) or inf (e5m2); the codecs and every
        // other backend clip to the largest finite value.
        if Fp8Format::of_type(ctx, to_ty).is_some() {
            conv.set_decoration_saturated_to_largest_float8_normal_conversion_ext(ctx);
        }
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
        } else if Fp8Format::of_type(scope.ctx(), input.scalar_ty(scope.ctx())).is_some() {
            scope.register_size::<N>(input.vector_size(scope.ctx()));
            let bytes = reinterpret_value(scope, input, Vector::<u8, N>::__expand_as_type(scope));
            fp8_to_bool::expand(scope, bytes.into()).read_value(scope)
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

/// Everything above the sign of an fp8 value, so that masking with it leaves the magnitude.
const FP8_MAGNITUDE_MASK: u8 = u8::MAX >> 1;

/// fp8 takes no comparison, so its bool conversion reads the bits. Masking the sign is what keeps
/// this a float test rather than a byte test: `-0.0` is `0x80`, whose magnitude is zero, so it
/// converts to `false` the way `0.0` does, while every NaN encoding has a non-zero magnitude and so
/// converts to `true`. Upstream CUDA documents the same rule for `__nv_fp8_e4m3::operator bool`:
/// "+0 and -0 inputs convert to `false`. Non-zero inputs convert to `true`."
#[cube]
fn fp8_to_bool(input: Vector<u8, N>) -> Vector<bool, N> {
    (input & Vector::new(FP8_MAGNITUDE_MASK)).not_equal(&Vector::zero())
}
