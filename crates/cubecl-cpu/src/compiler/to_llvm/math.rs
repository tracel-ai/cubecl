use super::ToLLVMDialect;
use cubecl_core::ir::dialect::math::*;
use cubecl_core::ir::prelude::*;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::utils::apint::APInt;
use pliron_llvm::attributes::FastmathFlagsAttr;
use pliron_llvm::op_interfaces::FloatBinArithOpWithFastMathFlags;
use pliron_llvm::types::{FuncType, VectorType, VectorTypeKind};
use pliron_llvm::{
    attributes::IntegerOverflowFlagsAttr, op_interfaces::IntBinArithOpWithOverflowFlag, ops as llvm,
};
use std::num::NonZero;

macro_rules! lower_unary_intrinsic_arith {
    ($cube_op:ty => $llvm_op:expr) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let input = self.input(ctx);
                let elem_ty = input.get_type(ctx);
                let intrinsic_type = FuncType::get(ctx, elem_ty, vec![elem_ty], false);

                let op =
                    llvm::CallIntrinsicOp::new(ctx, $llvm_op.into(), intrinsic_type, vec![input]);

                rewriter.insert_op(ctx, &op);
                rewriter.replace_operation_with_values(
                    ctx,
                    self.get_operation(),
                    vec![op.get_result(ctx)],
                );
                Ok(())
            }
        }
    };
}

lower_unary_intrinsic_arith!(FAbsOp => "llvm.fabs");
lower_unary_intrinsic_arith!(SAbsOp => "llvm.abs");
lower_unary_intrinsic_arith!(ExpOp => "llvm.exp");
lower_unary_intrinsic_arith!(LogOp => "llvm.log");
lower_unary_intrinsic_arith!(SinOp => "llvm.sin");
lower_unary_intrinsic_arith!(CosOp => "llvm.cos");
lower_unary_intrinsic_arith!(TanOp => "llvm.tan");
lower_unary_intrinsic_arith!(SinhOp => "llvm.sinh");
lower_unary_intrinsic_arith!(CoshOp => "llvm.cosh");
lower_unary_intrinsic_arith!(TanhOp => "llvm.tanh");
lower_unary_intrinsic_arith!(ArcSinOp => "llvm.asin");
lower_unary_intrinsic_arith!(ArcCosOp => "llvm.acos");
lower_unary_intrinsic_arith!(ArcTanOp => "llvm.atan");
lower_unary_intrinsic_arith!(SqrtOp => "llvm.sqrt");
lower_unary_intrinsic_arith!(RoundOp => "llvm.round");
lower_unary_intrinsic_arith!(FloorOp => "llvm.floor");
lower_unary_intrinsic_arith!(CeilOp => "llvm.ceil");
lower_unary_intrinsic_arith!(TruncOp => "llvm.trunc");

// See https://llvm.org/docs/LangRef.html#id1822 for more info
const IS_NAN: usize = 0x0003;
const IS_INF: usize = 0x0204;

macro_rules! lower_float_fpclass {
    ($cube_op:ty => $bitmask:expr) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let input = self.input(ctx);
                let elem_ty = input.get_type(ctx);
                let int_ty = IntegerType::get(ctx, 32, Signedness::Signless);
                let val = APInt::from_usize($bitmask, NonZero::new(32).unwrap());
                let constant_op = llvm::ConstantOp::new(ctx, IntegerAttr::new(int_ty, val).into());

                rewriter.insert_op(ctx, &constant_op);
                let val = constant_op.get_result(ctx);

                let mut bool_ty = IntegerType::get(ctx, 1, Signedness::Signless).into();
                if let Some(vector) = elem_ty.deref(ctx).downcast_ref::<VectorType>() {
                    let num_elems = vector.num_elements();
                    bool_ty =
                        VectorType::get(ctx, bool_ty, num_elems, VectorTypeKind::Fixed).into();
                }
                let intrinsic_type =
                    FuncType::get(ctx, bool_ty, vec![elem_ty, int_ty.into()], true);

                let op = llvm::CallIntrinsicOp::new(
                    ctx,
                    "llvm.is.fpclass".into(),
                    intrinsic_type,
                    vec![input, val],
                );

                rewriter.insert_op(ctx, &op);
                rewriter.replace_operation_with_values(
                    ctx,
                    self.get_operation(),
                    vec![op.get_result(ctx)],
                );
                Ok(())
            }
        }
    };
}

lower_float_fpclass!(IsNanOp => IS_NAN);
lower_float_fpclass!(IsInfOp => IS_INF);

macro_rules! lower_int_bin_arith {
    ($cube_op:ty => $llvm_op:ty) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let lhs = self.lhs(ctx);
                let rhs = self.rhs(ctx);
                let op = <$llvm_op>::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                );
                rewriter.insert_op(ctx, &op);
                rewriter.replace_operation_with_values(
                    ctx,
                    self.get_operation(),
                    vec![op.get_result(ctx)],
                );
                Ok(())
            }
        }
    };
}

lower_int_bin_arith!(IAddOp => llvm::AddOp);
lower_int_bin_arith!(IMulOp => llvm::MulOp);
lower_int_bin_arith!(ISubOp => llvm::SubOp);

macro_rules! lower_float_bin_arith {
    ($cube_op:ty => $llvm_op:ty) => {
        #[op_interface_impl]
        impl ToLLVMDialect for $cube_op {
            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                let lhs = self.lhs(ctx);
                let rhs = self.rhs(ctx);
                let op = <$llvm_op>::new_with_fast_math_flags(
                    ctx,
                    lhs,
                    rhs,
                    FastmathFlagsAttr::default(),
                );
                rewriter.insert_op(ctx, &op);
                rewriter.replace_operation_with_values(
                    ctx,
                    self.get_operation(),
                    vec![op.get_result(ctx)],
                );
                Ok(())
            }
        }
    };
}

lower_float_bin_arith!(FAddOp => llvm::FAddOp);
lower_float_bin_arith!(FSubOp => llvm::FSubOp);
lower_float_bin_arith!(FMulOp => llvm::FMulOp);
lower_float_bin_arith!(FDivOp => llvm::FDivOp);
lower_float_bin_arith!(FRemOp => llvm::FRemOp);

#[op_interface_impl]
impl ToLLVMDialect for FNegOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let op = llvm::FNegOp::new_with_fast_math_flags(ctx, input, FastmathFlagsAttr::default());
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}
