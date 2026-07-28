use super::prelude::*;
use cubecl_core::ir::attributes::{BoolAttr, FloatAttr, IndexAttr};
use half::f16;
use pliron::utils::apfloat::{self, Float};

/// Width LLVM expects for vector lane indices, intrinsic flags and `alloca` sizes.
pub const I32_WIDTH: u32 = 32;

/// Build the attribute of a `width`-bit signless integer holding `value`, truncated to `width`.
pub fn int_attr(ctx: &mut Context, width: u32, value: i128) -> IntegerAttr {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    IntegerAttr::new(ty, APInt::from_i128(value, bw(width as usize)))
}

/// Build the float attribute matching the float type `ty` and holding `value`, or `None` when `ty`
/// isn't a float type this backend supports.
pub fn float_attr(ctx: &Context, ty: TypeHandle, value: f64) -> Option<AttrObj> {
    Some(if ty.is_float16(ctx) {
        let value = f16::from_f64(value);
        FPHalfAttr(apfloat::Half::from_bits(value.to_bits() as u128)).into()
    } else if ty.is_float32(ctx) {
        FPSingleAttr::from(value as f32).into()
    } else if ty.is_float64(ctx) {
        FPDoubleAttr::from(value).into()
    } else {
        None?
    })
}

/// Insert a constant holding `value` as a `width`-bit signless integer, and return its result.
pub fn insert_int_const(
    ctx: &mut Context,
    rewriter: &mut impl Inserter,
    width: u32,
    value: i128,
) -> Value {
    let attr = int_attr(ctx, width, value);
    let op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_op(ctx, &op);
    op.get_result(ctx)
}

/// Insert an `i32` constant, and return its result.
pub fn insert_i32_const(ctx: &mut Context, rewriter: &mut impl Inserter, value: i32) -> Value {
    insert_int_const(ctx, rewriter, I32_WIDTH, value as i128)
}

/// Insert an `i1` constant, and return its result.
pub fn insert_bool_const(ctx: &mut Context, rewriter: &mut impl Inserter, value: bool) -> Value {
    insert_int_const(ctx, rewriter, 1, value as i128)
}

#[op_interface_impl]
impl ToLLVMDialect for llvm::ConstantOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.get_value(ctx);
        let result_ty = self.get_result(ctx).get_type(ctx);

        let const_value: AttrObj = if let Some(int) = value.downcast_ref::<IntegerAttr>() {
            let width = int.get_type().deref(ctx).width();
            IntegerAttr::new(
                IntegerType::get(ctx, width, Signedness::Signless),
                int.value(),
            )
            .into()
        } else if let Some(bool_attr) = value.downcast_ref::<BoolAttr>() {
            int_attr(ctx, 1, bool_attr.0 as i128).into()
        } else if let Some(index_attr) = value.downcast_ref::<IndexAttr>() {
            int_attr(ctx, INDEX_WIDTH, index_attr.0 as i128).into()
        } else if let Some(float) = value.downcast_ref::<FloatAttr>() {
            match float_attr(ctx, result_ty, apfloat::double_to_f64(float.val)) {
                Some(attr) => attr,
                None => return Ok(()),
            }
        } else {
            return Ok(());
        };

        let llvm_const = llvm::ConstantOp::new(ctx, const_value);
        rewriter.insert_operation(ctx, llvm_const.get_operation());

        let old_op = self.get_operation();
        rewriter.replace_operation(ctx, old_op, llvm_const.get_operation());

        Ok(())
    }
}
