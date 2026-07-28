use super::prelude::*;
use cubecl_core::ir::dialect::bitwise::*;
use cubecl_core::ir::dialect::cmp::{FMaxOp, FMinOp};
use cubecl_core::ir::dialect::general::{BoolAndOp, BoolNotOp, BoolOrOp};
use cubecl_core::ir::dialect::math::*;

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
                let res_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));
                let intrinsic_type = FuncType::get(ctx, res_ty, vec![elem_ty], false);

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
lower_unary_intrinsic_arith!(ReverseBitsOp => "llvm.bitreverse");

/// Width of an integer type, or of the elements of an integer vector type.
fn int_elem_width(ctx: &Context, ty: TypeHandle) -> u32 {
    let elem_ty = ty
        .deref(ctx)
        .downcast_ref::<LlvmVectorType>()
        .map(|vector| vector.elem_type())
        .unwrap_or(ty);
    elem_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .expect("bit counting intrinsics only apply to integers")
        .width()
}

macro_rules! lower_count_bits_intrinsic {
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
                let res_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));

                let params = vec![elem_ty];
                let args = vec![input];
                let intrinsic_type = FuncType::get(ctx, elem_ty, params, false);

                let op = llvm::CallIntrinsicOp::new(ctx, $llvm_op.into(), intrinsic_type, args);
                rewriter.insert_op(ctx, &op);

                let count = op.get_result(ctx);
                let in_width = int_elem_width(ctx, elem_ty);
                let out_width = int_elem_width(ctx, res_ty);
                let count = if in_width == out_width {
                    count
                } else if in_width > out_width {
                    let trunc = llvm::TruncOp::new(ctx, count, res_ty);
                    rewriter.insert_op(ctx, &trunc);
                    trunc.get_result(ctx)
                } else {
                    let zext = llvm::ZExtOp::new_with_nneg(ctx, count, res_ty, false);
                    rewriter.insert_op(ctx, &zext);
                    zext.get_result(ctx)
                };

                rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![count]);
                Ok(())
            }
        }
    };
}

lower_count_bits_intrinsic!(CountOnesOp => "llvm.ctpop");
lower_count_bits_intrinsic!(LeadingZerosBitsOp => "llvm.ctlz");
lower_count_bits_intrinsic!(TrailingZerosBitsOp => "llvm.cttz");

// See https://llvm.org/docs/LangRef.html#id1822 for more info
const IS_NAN: i32 = 0x0003;
const IS_INF: i32 = 0x0204;

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
                let int_ty = IntegerType::get(ctx, I32_WIDTH, Signedness::Signless);
                let val = insert_i32_const(ctx, rewriter, $bitmask);

                let mut bool_ty = IntegerType::get(ctx, 1, Signedness::Signless).into();
                if let Some(vector) = elem_ty.deref(ctx).downcast_ref::<LlvmVectorType>() {
                    let num_elems = vector.num_elements();
                    bool_ty =
                        LlvmVectorType::get(ctx, bool_ty, num_elems, VectorTypeKind::Fixed).into();
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

macro_rules! lower_int_bin_with_overflow_arith {
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

lower_int_bin_with_overflow_arith!(IAddOp => llvm::AddOp);
lower_int_bin_with_overflow_arith!(IMulOp => llvm::MulOp);
lower_int_bin_with_overflow_arith!(ISubOp => llvm::SubOp);

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
                let op = <$llvm_op>::new(ctx, lhs, rhs);
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

lower_int_bin_arith!(BoolAndOp => llvm::AndOp);
lower_int_bin_arith!(BoolOrOp => llvm::OrOp);

lower_int_bin_arith!(BitwiseAndOp => llvm::AndOp);
lower_int_bin_arith!(BitwiseOrOp => llvm::OrOp);
lower_int_bin_arith!(BitwiseXorOp => llvm::XorOp);
lower_int_bin_arith!(UDivOp => llvm::UDivOp);
lower_int_bin_arith!(URemOp => llvm::URemOp);

#[op_interface_impl]
impl ToLLVMDialect for ShiftRightOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let original_lhs_ty = *operands_info.lookup_operand_history(lhs).first().unwrap();
        let op: &dyn OneResultInterface = if original_lhs_ty.is_signed_int(ctx) {
            &llvm::AShrOp::new(ctx, lhs, rhs)
        } else {
            &llvm::LShrOp::new(ctx, lhs, rhs)
        };
        rewriter.insert_op(ctx, op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

lower_int_bin_arith!(ShiftLeftOp => llvm::ShlOp);

// LLVM has no boolean negation, so `!x` becomes `x ^ true`.
#[op_interface_impl]
impl ToLLVMDialect for BoolNotOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let input = self.input(ctx);
        let res_ty = self.get_result(ctx).get_type(ctx);
        let num_lanes = res_ty.vector_size(ctx);

        let mut ones = insert_bool_const(ctx, rewriter, true);
        if num_lanes > 1 {
            let vec_ty = cube_type_to_llvm(ctx, res_ty);
            ones = insert_splat(ctx, rewriter, vec_ty, ones, num_lanes);
        }

        let op = llvm::XorOp::new(ctx, input, ones);
        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}

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

macro_rules! lower_binary_intrinsic_arith {
    ($cube_op:ty => $llvm_op:expr) => {
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
                let lhs_ty = lhs.get_type(ctx);
                let rhs_ty = rhs.get_type(ctx);
                let res_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));
                let intrinsic_type = FuncType::get(ctx, res_ty, vec![lhs_ty, rhs_ty], false);

                let op = llvm::CallIntrinsicOp::new(
                    ctx,
                    $llvm_op.into(),
                    intrinsic_type,
                    vec![lhs, rhs],
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

lower_binary_intrinsic_arith!(ArcTan2Op => "llvm.atan2");
lower_binary_intrinsic_arith!(PowfOp => "llvm.pow");
lower_binary_intrinsic_arith!(FMinOp => "llvm.minnum");
lower_binary_intrinsic_arith!(FMaxOp => "llvm.maxnum");

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

#[op_interface_impl]
impl ToLLVMDialect for FmaOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let a = self.a(ctx);
        let b = self.b(ctx);
        let c = self.c(ctx);
        let a_ty = a.get_type(ctx);
        let b_ty = b.get_type(ctx);
        let c_ty = c.get_type(ctx);
        let res_ty = cube_type_to_llvm(ctx, self.get_result(ctx).get_type(ctx));
        let intrinsic_type = FuncType::get(ctx, res_ty, vec![a_ty, b_ty, c_ty], false);

        let op =
            llvm::CallIntrinsicOp::new(ctx, "llvm.fmuladd".into(), intrinsic_type, vec![a, b, c]);

        rewriter.insert_op(ctx, &op);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![op.get_result(ctx)]);
        Ok(())
    }
}
