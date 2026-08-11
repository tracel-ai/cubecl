use cubecl_core::prelude::polyfills::bitwise::*;
use cubecl_ir::{dialect::bitwise::*, interfaces::TypedExt, prelude::*, types::VectorType};
use pliron::builtin::types::{IntegerType, Signedness};

use crate::compiler::wgsl::{
    lower::lower_unop,
    to_wgsl::{TypeExtWgsl, wgsl_op_with_out},
};

wgsl_op_with_out!(BitwiseAndOp; |op, ctx| {
    format!("{} & {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(BitwiseOrOp; |op, ctx| {
    format!("{} | {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(BitwiseXorOp; |op, ctx| {
    format!("{} ^ {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(ShiftLeftOp; |op, ctx| {
    let u32 = u32_ty(ctx, op.lhs(ctx)).to_wgsl(ctx);
    format!("{} << {u32}({})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(ShiftRightOp; |op, ctx| {
    let u32 = u32_ty(ctx, op.lhs(ctx)).to_wgsl(ctx);
    format!("{} >> {u32}({})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

wgsl_op_with_out!(BitwiseNotOp; |op, ctx| {
    format!("~{}", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(CountOnesOp; |op, ctx| {
    let u32 = u32_ty(ctx, op.input(ctx)).to_wgsl(ctx);
    format!("{u32}(countOneBits({}))", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(ReverseBitsOp; |op, ctx| {
    format!("reverseBits({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(LeadingZerosBitsOp; |op, ctx| {
    let u32 = u32_ty(ctx, op.input(ctx)).to_wgsl(ctx);
    format!("{u32}(countLeadingZeros({}))", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(TrailingZerosBitsOp; |op, ctx| {
    let u32 = u32_ty(ctx, op.input(ctx)).to_wgsl(ctx);
    format!("{u32}(countTrailingZeros({}))", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(FindFirstSetOp; |op, ctx| {
    let u32 = u32_ty(ctx, op.input(ctx)).to_wgsl(ctx);
    format!("{u32}(firstTrailingBit({})) + {u32}(1)", op.input(ctx).name(ctx))
});

lower_unop!(LeadingZerosBitsOp, u64_leading_zeros, |op, ctx| {
    op.input(ctx).scalar_ty(ctx).is_int_of_width(ctx, 64)
});
lower_unop!(TrailingZerosBitsOp, u64_trailing_zeros, |op, ctx| {
    op.input(ctx).scalar_ty(ctx).is_int_of_width(ctx, 64)
});
lower_unop!(FindFirstSetOp, u64_ffs, |op, ctx| {
    op.input(ctx).scalar_ty(ctx).is_int_of_width(ctx, 64)
});

fn u32_ty(ctx: &Context, reference_ty: impl Typed) -> TypeHandle {
    let vec = reference_ty.vector_size(ctx);
    let u32 = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
    if vec > 1 {
        VectorType::get(ctx, u32, vec).to_handle()
    } else {
        u32
    }
}
