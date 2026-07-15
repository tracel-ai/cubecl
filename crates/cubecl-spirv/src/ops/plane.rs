use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{dialect::plane, interfaces::TypedExt, prelude::*, types::scalar::BoolType};
use pliron::builtin::ops::ConstantOp;
use pliron_spirv::ops::{self};
use rspirv::spirv::{GroupOperation, MemoryAccess, Scope};

use crate::{
    lower::LowerOp,
    ops::{atomic::semantics_r, to_spirv_dialect::ToSpirvDialectOp},
    types::ty_to_spirv_dialect,
};

#[op_interface_impl]
impl ToSpirvDialectOp for plane::ElectOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let out_ty = ty_to_spirv_dialect(ctx, BoolType::get(ctx));
        let new_op = ops::GroupNonUniformElectOp::new(ctx, out_ty, Scope::Subgroup);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

macro_rules! plane_unop_to_spirv_dialect {
    ($ty: ty => $new_ty: ty $(,$extra:expr)*) => {
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
                let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, Scope::Subgroup, inp);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}

macro_rules! plane_binop_to_spirv_dialect {
    ($ty: ty => $new_ty: ty $(,$extra:expr)*) => {
        #[op_interface_impl]
        impl ToSpirvDialectOp for $ty {
            fn to_spirv_dialect(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let op = self.get_operation();
                let value = op.operand(ctx, 0);
                let lane = op.operand(ctx, 1);
                let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, Scope::Subgroup, value, lane);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}

macro_rules! plane_reduce_op_to_spirv_dialect {
    ($ty: ty => $new_ty: ty, $action: expr $(,$extra:expr)*) => {
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
                let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
                let new_op = <$new_ty>::new(ctx, out_ty, Scope::Subgroup, $action, inp, None);
                rewriter.append_op(ctx, &new_op);
                rewriter.replace_operation(ctx, op, new_op.get_operation());

                Ok(())
            }
        }
    };
}

plane_binop_to_spirv_dialect!(plane::ShuffleOp => ops::GroupNonUniformShuffleOp);
plane_binop_to_spirv_dialect!(plane::ShuffleXorOp => ops::GroupNonUniformShuffleXorOp);
plane_binop_to_spirv_dialect!(plane::ShuffleUpOp => ops::GroupNonUniformShuffleUpOp);
plane_binop_to_spirv_dialect!(plane::ShuffleDownOp => ops::GroupNonUniformShuffleDownOp);

plane_unop_to_spirv_dialect!(plane::AllOp => ops::GroupNonUniformAllOp);
plane_unop_to_spirv_dialect!(plane::AnyOp => ops::GroupNonUniformAnyOp);
plane_unop_to_spirv_dialect!(plane::BallotOp => ops::GroupNonUniformBallotOp);

plane_reduce_op_to_spirv_dialect!(plane::ISumOp => ops::GroupNonUniformIAddOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::FSumOp => ops::GroupNonUniformFAddOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::InclusiveISumOp => ops::GroupNonUniformIAddOp, GroupOperation::InclusiveScan);
plane_reduce_op_to_spirv_dialect!(plane::InclusiveFSumOp => ops::GroupNonUniformFAddOp, GroupOperation::InclusiveScan);
plane_reduce_op_to_spirv_dialect!(plane::ExclusiveISumOp => ops::GroupNonUniformIAddOp, GroupOperation::ExclusiveScan);
plane_reduce_op_to_spirv_dialect!(plane::ExclusiveFSumOp => ops::GroupNonUniformFAddOp, GroupOperation::ExclusiveScan);

plane_reduce_op_to_spirv_dialect!(plane::IProdOp => ops::GroupNonUniformIMulOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::FProdOp => ops::GroupNonUniformFMulOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::InclusiveIProdOp => ops::GroupNonUniformIMulOp, GroupOperation::InclusiveScan);
plane_reduce_op_to_spirv_dialect!(plane::InclusiveFProdOp => ops::GroupNonUniformFMulOp, GroupOperation::InclusiveScan);
plane_reduce_op_to_spirv_dialect!(plane::ExclusiveIProdOp => ops::GroupNonUniformIMulOp, GroupOperation::ExclusiveScan);
plane_reduce_op_to_spirv_dialect!(plane::ExclusiveFProdOp => ops::GroupNonUniformFMulOp, GroupOperation::ExclusiveScan);

plane_reduce_op_to_spirv_dialect!(plane::SMinOp => ops::GroupNonUniformSMinOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::UMinOp => ops::GroupNonUniformUMinOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::FMinOp => ops::GroupNonUniformFMinOp, GroupOperation::Reduce);

plane_reduce_op_to_spirv_dialect!(plane::SMaxOp => ops::GroupNonUniformSMaxOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::UMaxOp => ops::GroupNonUniformUMaxOp, GroupOperation::Reduce);
plane_reduce_op_to_spirv_dialect!(plane::FMaxOp => ops::GroupNonUniformFMaxOp, GroupOperation::Reduce);

#[op_interface_impl]
impl ToSpirvDialectOp for plane::BroadcastOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let value = self.input(ctx);
        let lane = *self.lane(ctx);
        let lane_const = ConstantOp::new(ctx, lane.into());
        rewriter.append_op(ctx, &lane_const);
        let lane = lane_const.get_result(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let new_op =
            ops::GroupNonUniformBroadcastOp::new(ctx, out_ty, Scope::Subgroup, value, lane);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for plane::UniformLoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let ptr = self.ptr(ctx);
        let align = self.result_type(ctx).align(ctx) as u32;
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let new_op = ops::LoadOp::new(ctx, out_ty, ptr, MemoryAccess::ALIGNED, Some(align));
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for plane::AtomicUniformLoadOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let ptr = self.ptr(ctx);
        let semantics = semantics_r(ctx, ptr);
        let out_ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        let new_op = ops::AtomicLoadOp::new(ctx, out_ty, ptr, Scope::Subgroup, semantics);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());
        Ok(())
    }
}

define_scalar!(T);
define_size!(N);

macro_rules! unroll_plane_unop {
    ($ty: ty, $s_ty: ident, $op: expr) => {
        const _: () = {
            #[cube]
            fn unroll(input: Vector<$s_ty, N>) -> Vector<$s_ty, N> {
                let mut out = Vector::default();
                #[unroll]
                for i in 0..input.vector_size() {
                    out.insert(i, $op(input.extract(i)));
                }
                out
            }

            #[op_interface_impl]
            impl LowerOp for $ty {
                fn should_lower(&self, ctx: &Context) -> bool {
                    self.get_operand(ctx).vector_size(ctx) > 1
                }

                fn lower(&self, scope: &cubecl_ir::Scope) -> Vec<Value> {
                    let inp = self.get_operand(scope.ctx());
                    scope.register_value_type::<T, N>(inp);
                    vec![unroll::expand(scope, inp.into()).read_value(scope)]
                }
            }
        };
    };
}

unroll_plane_unop!(plane::AllOp, bool, plane_all);
unroll_plane_unop!(plane::AnyOp, bool, plane_any);
