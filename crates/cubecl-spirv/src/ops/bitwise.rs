use cubecl_core::{self as cubecl, WgpuCompilationOptions, num_traits::Zero, prelude::*};
use cubecl_ir::{
    dialect::bitwise::{self, *},
    interfaces::TypedExt,
    prelude::*,
};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    utils::apint::{APInt, bw},
};
use pliron_spirv::{
    ext::gl,
    ops::{self, IAddOp, ISubOp, ShiftRightArithmeticOp, ShiftRightLogicalOp},
};

use crate::{
    lower::lower_unop,
    ops::{
        base::{binop_to_spirv_dialect, unop_to_spirv_dialect},
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

define_size!(N);

binop_to_spirv_dialect!(bitwise::BitwiseAndOp => ops::BitwiseAndOp);
binop_to_spirv_dialect!(bitwise::BitwiseOrOp => ops::BitwiseOrOp);
binop_to_spirv_dialect!(bitwise::BitwiseXorOp => ops::BitwiseXorOp);
binop_to_spirv_dialect!(bitwise::ShiftLeftOp => ops::ShiftLeftLogicalOp);

unop_to_spirv_dialect!(bitwise::BitwiseNotOp => ops::NotOp);
unop_to_spirv_dialect!(bitwise::CountOnesOp => ops::BitCountOp);
unop_to_spirv_dialect!(bitwise::ReverseBitsOp => ops::BitReverseOp);

#[op_interface_impl]
impl ToSpirvDialectOp for bitwise::ShiftRightOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let lhs = op.operand(ctx, 0);
        let rhs = op.operand(ctx, 1);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        if self.get_result(ctx).scalar_ty(ctx).is_signed_int(ctx) {
            let new_op = ShiftRightArithmeticOp::new(ctx, out_ty, lhs, rhs);
            rewriter.append_op(ctx, &new_op);
            rewriter.replace_operation(ctx, op, new_op.get_operation());
        } else {
            let new_op = ShiftRightLogicalOp::new(ctx, out_ty, lhs, rhs);
            rewriter.append_op(ctx, &new_op);
            rewriter.replace_operation(ctx, op, new_op.get_operation());
        }

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for bitwise::LeadingZerosBitsOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let op = self.get_operation();
        let inp = self.input(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));

        // Indices are zero based, so subtract 1 from u32 width
        let width = const_u32(&scope, 31);
        let msb = if self.get_result(ctx).scalar_ty(ctx).is_signed_int(ctx) {
            scope.register_with_result(&gl::FindSMsbOp::new(ctx, out_ty, inp))
        } else {
            scope.register_with_result(&gl::FindUMsbOp::new(ctx, out_ty, inp))
        };
        let value = scope.register_with_result(&ISubOp::new(ctx, out_ty, msb, width));
        rewriter.replace_operation_with_values(ctx, op, vec![value]);

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for bitwise::FindFirstSetOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let op = self.get_operation();
        let inp = self.input(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));

        let one = const_u32(&scope, 1);
        let lsb = scope.register_with_result(&gl::FindILsbOp::new(ctx, out_ty, inp));
        // Normalize to CUDA/POSIX convention of 1 based index, with 0 meaning not found
        let value = scope.register_with_result(&IAddOp::new(ctx, out_ty, lsb, one));
        rewriter.replace_operation_with_values(ctx, op, vec![value]);

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for bitwise::TrailingZerosBitsOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let scope = Scope::from_context_and_inserter(ctx, rewriter);
        let op = self.get_operation();
        let inp = self.input(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));

        let lsb = scope.register_with_result(&gl::FindILsbOp::new(ctx, out_ty, inp));
        scope.register_value_type::<(), N>(inp);
        let value = trailing_zeros_adjust::expand(&scope, inp.into(), lsb.into()).value(&scope);
        rewriter.replace_operation_with_values(ctx, op, vec![value]);

        Ok(())
    }
}

fn const_u32(scope: &Scope, value: u32) -> Value {
    let u32 = IntegerType::get(scope.ctx(), 32, Signedness::Signless);
    let value = IntegerAttr::new(u32, APInt::from_u32(value, bw(32)));
    scope.register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(value)))
}

lower_unop!(CountOnesOp, u64_count_bits, |op, ctx| {
    op.get_result(ctx).is_int_of_width(ctx, 64) && !has_u64_bitwise(ctx)
});
lower_unop!(ReverseBitsOp, u64_reverse, |op, ctx| {
    op.get_result(ctx).is_int_of_width(ctx, 64) && !has_u64_bitwise(ctx)
});
lower_unop!(LeadingZerosBitsOp, u64_leading_zeros, |op, ctx| {
    op.get_result(ctx).is_int_of_width(ctx, 64)
});
lower_unop!(FindFirstSetOp, u64_ffs, |op, ctx| {
    op.get_result(ctx).is_int_of_width(ctx, 64)
});
lower_unop!(TrailingZerosBitsOp, u64_trailing_zeros, |op, ctx| {
    op.get_result(ctx).is_int_of_width(ctx, 64)
});

fn has_u64_bitwise(ctx: &Context) -> bool {
    ctx.aux_ty::<WgpuCompilationOptions>()
        .vulkan
        .supports_arbitrary_bitwise
}

#[cube]
fn u64_reverse<I: Int, N: Size>(x: Vector<I, N>) -> Vector<I, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);

    let low_rev = Vector::reverse_bits(low);
    let high_rev = Vector::reverse_bits(high);
    // Swap low and high values
    let high = Vector::cast_from(low_rev) << shift;
    high | Vector::cast_from(high_rev)
}

#[cube]
fn u64_count_bits<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);

    let low_cnt = Vector::<u32, N>::cast_from(Vector::count_ones(low));
    let high_cnt = Vector::<u32, N>::cast_from(Vector::count_ones(high));
    low_cnt + high_cnt
}

#[cube]
fn u64_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);
    let low_zeros = Vector::leading_zeros(low);
    let high_zeros = Vector::leading_zeros(high);

    select_many(
        high_zeros.equal(&Vector::new(32)),
        low_zeros + high_zeros,
        high_zeros,
    )
}

/// There are three possible outcomes:
/// * low has any set -> return low
/// * low is empty, high has any set -> return high + 32
/// * low and high are empty -> return 0
#[cube]
fn u64_ffs<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);
    let low_ffs = Vector::find_first_set(low);
    let high_ffs = Vector::find_first_set(high);

    let high_ffs = select_many(
        high_ffs.equal(&Vector::new(0)),
        high_ffs,
        high_ffs + Vector::new(32),
    );
    select_many(low_ffs.equal(&Vector::new(0)), high_ffs, low_ffs)
}

/// There are three possible outcomes:
/// * low has any set -> return low
/// * low is empty, high has any set -> return high + 32
/// * low and high are empty -> return 0
#[cube]
fn u64_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
    let shift = Vector::new(I::new(32));

    let low = Vector::<u32, N>::cast_from(x);
    let high = Vector::<u32, N>::cast_from(x >> shift);
    let low_tz = Vector::trailing_zeros(low);
    let high_tz = Vector::trailing_zeros(high);

    let high_tz = select_many(
        high_tz.equal(&Vector::new(32)),
        Vector::new(64),
        high_tz + Vector::new(32),
    );
    select_many(low_tz.equal(&Vector::new(32)), high_tz, low_tz)
}

// find_lsb returns -1 (0xFFFFFFFF) for zero input
// trailing_zeros should return bit_width for zero input
#[cube]
fn trailing_zeros_adjust(inp: Vector<u32, N>, lsb: Vector<u32, N>) -> Vector<u32, N> {
    let width = Vector::new(32);
    select_many(inp.equal(&Vector::zero()), width, lsb)
}
