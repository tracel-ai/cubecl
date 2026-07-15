use cubecl_ir::{dialect::cmp, prelude::*};
use pliron::irbuild::inserter::Inserter;
use pliron_spirv::{ext::gl, ops};

use crate::{
    ops::{base::binop_to_spirv_dialect, to_spirv_dialect::ToSpirvDialectOp},
    types::ty_to_spirv_dialect,
};

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
