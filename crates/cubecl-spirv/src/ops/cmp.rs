use cubecl_core::{self as cubecl, define_size, prelude::*};
use cubecl_ir::{
    Scope,
    dialect::{base::OperationPtrExt, cmp},
    interfaces::TypedExt,
    prelude::*,
    types::Fp8Format,
};
use pliron::irbuild::inserter::Inserter;
use pliron_spirv::{ext::gl, ops};

use crate::{
    lower::LowerOp,
    ops::{base::binop_to_spirv_dialect, to_spirv_dialect::ToSpirvDialectOp},
    types::ty_to_spirv_dialect,
};

define_size!(N);

macro_rules! clamp_to_spirv_dialect {
    ($ty: ty => $new_ty: ty) => {
        #[op_interface_impl]
        impl ToSpirvDialectOp for $ty {
            fn to_spirv_dialect(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let op = self.get_operation();
                let inp = op.operand(ctx, 0);
                let min = op.operand(ctx, 1);
                let max = op.operand(ctx, 2);
                let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, inp, min, max);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}

binop_to_spirv_dialect!(cmp::SMinOp => gl::SMinOp);
binop_to_spirv_dialect!(cmp::UMinOp => gl::UMinOp);
binop_to_spirv_dialect!(cmp::FMinOp => gl::FMinOp);

binop_to_spirv_dialect!(cmp::SMaxOp => gl::SMaxOp);
binop_to_spirv_dialect!(cmp::UMaxOp => gl::UMaxOp);
binop_to_spirv_dialect!(cmp::FMaxOp => gl::FMaxOp);

clamp_to_spirv_dialect!(cmp::SClampOp => gl::SClampOp);
clamp_to_spirv_dialect!(cmp::UClampOp => gl::UClampOp);
clamp_to_spirv_dialect!(cmp::FClampOp => gl::FClampOp);

binop_to_spirv_dialect!(cmp::IEqualOp => ops::IEqualOp);
binop_to_spirv_dialect!(cmp::FEqualOp => ops::FOrdEqualOp);

binop_to_spirv_dialect!(cmp::INotEqualOp => ops::INotEqualOp);
binop_to_spirv_dialect!(cmp::FNotEqualOp => ops::FOrdNotEqualOp);

binop_to_spirv_dialect!(cmp::SGreaterThanOp => ops::SGreaterThanOp);
binop_to_spirv_dialect!(cmp::UGreaterThanOp => ops::UGreaterThanOp);
binop_to_spirv_dialect!(cmp::FGreaterThanOp => ops::FOrdGreaterThanOp);

binop_to_spirv_dialect!(cmp::SGreaterThanOrEqualOp => ops::SGreaterThanEqualOp);
binop_to_spirv_dialect!(cmp::UGreaterThanOrEqualOp => ops::UGreaterThanEqualOp);
binop_to_spirv_dialect!(cmp::FGreaterThanOrEqualOp => ops::FOrdGreaterThanEqualOp);

binop_to_spirv_dialect!(cmp::SLessThanOp => ops::SLessThanOp);
binop_to_spirv_dialect!(cmp::ULessThanOp => ops::ULessThanOp);
binop_to_spirv_dialect!(cmp::FLessThanOp => ops::FOrdLessThanOp);

binop_to_spirv_dialect!(cmp::SLessThanOrEqualOp => ops::SLessThanEqualOp);
binop_to_spirv_dialect!(cmp::ULessThanOrEqualOp => ops::ULessThanEqualOp);
binop_to_spirv_dialect!(cmp::FLessThanOrEqualOp => ops::FOrdLessThanEqualOp);

/// Neither spelling of fp8 can take a float comparison. `VK_EXT_shader_float8` allows conversion,
/// cooperative matrix multiply, and the operations that only move bits around, and the 8-bit
/// integer that stands in for fp8 without it is not a float type at all. Both compare on the bits,
/// which is also what a CUDA kernel gets: `__nv_fp8_e4m3` declares no comparison operators, and
/// this project's CUDA backend stores fp8 as the raw `__nv_fp8_storage_t` byte.
///
/// Bit equality parts from float equality in two places, the same two: `0.0` and `-0.0` are equal
/// as floats but not as bits, and a NaN equals itself here where a float NaN does not. Scale
/// factors, the one place fp8 sees much use, are non-negative and never NaN, so neither case
/// reaches them.
macro_rules! fp8_compares_bits {
    ($ty: ty, $name: ident) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                let lhs = self.get_operation().operand(ctx, 0);
                Fp8Format::of_type(ctx, lhs.scalar_ty(ctx)).is_some()
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let lhs = self.get_operation().operand(scope.ctx(), 0);
                let rhs = self.get_operation().operand(scope.ctx(), 1);
                scope.register_size::<N>(lhs.vector_size(scope.ctx()));
                let bytes_ty = Vector::<u8, N>::__expand_as_type(scope);
                let lhs = reinterpret_value(scope, lhs, bytes_ty);
                let rhs = reinterpret_value(scope, rhs, bytes_ty);
                vec![$name::expand(scope, lhs.into(), rhs.into()).read_value(scope)]
            }
        }
    };
}

#[cube]
fn bits_equal(lhs: Vector<u8, N>, rhs: Vector<u8, N>) -> Vector<bool, N> {
    lhs.equal(&rhs)
}

#[cube]
fn bits_not_equal(lhs: Vector<u8, N>, rhs: Vector<u8, N>) -> Vector<bool, N> {
    lhs.not_equal(&rhs)
}

fp8_compares_bits!(cmp::FEqualOp, bits_equal);
fp8_compares_bits!(cmp::FNotEqualOp, bits_not_equal);
