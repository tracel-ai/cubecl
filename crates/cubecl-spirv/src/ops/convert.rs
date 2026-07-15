use cubecl_core::{
    self as cubecl, define_scalar, define_size,
    num_traits::{One, Zero},
    prelude::*,
};
use cubecl_ir::{
    CanMaterialize, Pure, Scope,
    dialect::general,
    interfaces::{TriviallyUnrollable, TypedExt},
    prelude::*,
};
use pliron::printable::Printable;
use pliron_spirv::{ops, types::FloatType};

use crate::{
    lower::LowerOp,
    ops::{base::unop_to_spirv_dialect, to_spirv_dialect::ToSpirvDialectOp},
    types::ty_to_spirv_dialect,
};

macro_rules! cast_op {
    ($name: literal, $ty: ident $(=> $spirv_ty: ty)*) => {
        #[cube_op(name = $name)]
        #[result_ty(argument)]
        #[op_interfaces(SameOperandsType, TriviallyUnrollable)]
        #[op_traits(CanMaterialize, Pure)]
        pub struct $ty {
            pub input: Value,
        }
        $(unop_to_spirv_dialect!($ty => $spirv_ty);)*
    };
}

cast_op!("cube.spirv.convert_nop", ConvertNopOp);
cast_op!("cube.spirv.convert_f_to_u", ConvertFToUOp => ops::ConvertFToUOp);
cast_op!("cube.spirv.convert_f_to_s", ConvertFToSOp => ops::ConvertFToSOp);
cast_op!("cube.spirv.convert_u_to_f", ConvertUToFOp => ops::ConvertUToFOp);
cast_op!("cube.spirv.convert_s_to_f", ConvertSToFOp => ops::ConvertSToFOp);
cast_op!("cube.spirv.u_convert", UConvertOp => ops::UConvertOp);
cast_op!("cube.spirv.s_convert", SConvertOp => ops::SConvertOp);
cast_op!("cube.spirv.f_convert", FConvertOp => ops::FConvertOp);

#[op_interface_impl]
impl ToSpirvDialectOp for ConvertNopOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _: &OperandsInfo,
    ) -> Result<()> {
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![self.input(ctx)]);
        Ok(())
    }
}

define_scalar!(TIn);
define_scalar!(TOut);
define_size!(N);

// Lower this ahead of time when we still have signedness info. Conversion to SPIR-V dialect erases
// the sign.
#[op_interface_impl]
impl LowerOp for general::CastOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let input = self.input(scope.ctx());
        let to = self.result_type(scope.ctx());
        scope.register_value_type::<TIn, N>(input);
        scope.register_value_type::<TOut, ()>(to);
        vec![cast(scope, input, to)]
    }
}

fn cast(scope: &Scope, from: Value, to: TypeHandle) -> Value {
    let ctx = scope.ctx_mut();
    let in_ty = from.get_type(ctx).scalar_ty(ctx);
    let to_ty = to.get_type(ctx).scalar_ty(ctx);
    let in_ty_spirv = ty_to_spirv_dialect(ctx, in_ty);
    let to_ty_spirv = ty_to_spirv_dialect(ctx, to_ty);

    let in_float = in_ty_spirv.deref(ctx).is::<FloatType>();
    let in_sint = in_ty.is_signed_int(ctx);
    let in_uint = in_ty.is_unsigned_int(ctx) || in_ty.is_index(ctx);
    let out_float = to_ty_spirv.deref(ctx).is::<FloatType>();
    let out_sint = to_ty.is_signed_int(ctx);
    let out_uint = to_ty.is_unsigned_int(ctx) || to_ty.is_index(ctx);

    if in_ty_spirv == to_ty_spirv {
        scope.register_with_result(&ConvertNopOp::new(ctx, to, from))
    } else if in_float && out_uint {
        scope.register_with_result(&ConvertFToUOp::new(ctx, to, from))
    } else if in_float && out_sint {
        scope.register_with_result(&ConvertFToSOp::new(ctx, to, from))
    } else if in_sint && out_float {
        scope.register_with_result(&ConvertSToFOp::new(ctx, to, from))
    } else if in_uint && out_float {
        scope.register_with_result(&ConvertUToFOp::new(ctx, to, from))
    } else if in_uint && (to_ty.is_int(ctx) || to_ty.is_index(ctx)) {
        scope.register_with_result(&UConvertOp::new(ctx, to, from))
    } else if in_sint && (to_ty.is_int(ctx) || to_ty.is_index(ctx)) {
        scope.register_with_result(&SConvertOp::new(ctx, to, from))
    } else if in_float && out_float {
        scope.register_with_result(&FConvertOp::new(ctx, to, from))
    } else if in_ty.is_bool(ctx) {
        bool_to_numeric::expand::<TOut>(scope, from.into()).read_value(scope)
    } else if to_ty.is_bool(ctx) {
        numeric_to_bool::expand::<TIn>(scope, from.into()).read_value(scope)
    } else {
        panic!(
            "cast from {} to {} not supported",
            from.get_type(ctx).disp(ctx),
            to.disp(ctx)
        )
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
