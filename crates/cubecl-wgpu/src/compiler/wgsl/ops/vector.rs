use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{dialect::vector::*, interfaces::TypedExt, prelude::*};
use itertools::Itertools;

use crate::compiler::wgsl::{
    lower::lower_unop,
    to_wgsl::{TypeExtWgsl, wgsl_op, wgsl_op_with_out},
    value::WgslValue,
};

wgsl_op_with_out!(VectorInitOp, |op, ctx| {
    let ty = op.result_type(ctx).to_wgsl(ctx);
    let mut values = op.values(ctx).into_iter().map(|val| val.name(ctx));
    format!("{ty}({})", values.join(", "))
});

wgsl_op_with_out!(VectorBroadcastOp, |op, ctx| {
    let ty = op.result_type(ctx).to_wgsl(ctx);
    format!("{ty}({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(VectorInsertOp, |op, ctx| {
    let ty = op.result_type(ctx).to_wgsl(ctx);
    let vec = op.result_type(ctx).vector_size(ctx);
    let idx = op.index(ctx).0;
    let vector = op.vector(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    let mut values = (0..vec).map(|i| {
        if i == idx {
            value.to_string()
        } else {
            format!("{vector}[{i}]")
        }
    });
    format!("{ty}({})", values.join(", "))
});

wgsl_op_with_out!(VectorExtractOp, |op, ctx| {
    let idx = op.index(ctx).0;
    format!("{}[{idx}]", op.vector(ctx).name(ctx))
});

wgsl_op!(VectorInsertDynamicOp, |op, ctx| {
    let out = op.get_result(ctx).name(ctx);
    let ty = op.result_type(ctx).to_wgsl(ctx);
    let vector = op.vector(ctx).name(ctx);
    let index = op.index(ctx).name(ctx);
    let value = op.value(ctx).name(ctx);
    format!(
        "
    var {out}_tmp: {ty} = {vector};
    {out}_tmp[{index}] = {value};
    {} = {out}_tmp;",
        op.get_result(ctx).fmt_left(ctx)
    )
});

wgsl_op_with_out!(VectorExtractDynamicOp, |op, ctx| {
    let idx = op.index(ctx).name(ctx);
    format!("{}[{idx}]", op.vector(ctx).name(ctx))
});

wgsl_op_with_out!(MagnitudeOp, |op, ctx| {
    format!("length({})", op.input(ctx).name(ctx))
});
wgsl_op_with_out!(NormalizeOp, |op, ctx| {
    format!("normalize({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SDotOp, |op, ctx| {
    format!("dot({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(UDotOp, |op, ctx| {
    format!("dot({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(FDotOp, |op, ctx| {
    format!("dot({}, {})", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});

lower_unop!(ISumOp, sum);
lower_unop!(FSumOp, sum);

#[cube]
fn sum<T: Numeric, N: Size>(input: Vector<T, N>) -> T {
    let mut sum = T::from_int(0);
    #[unroll]
    for i in 0..input.vector_size() {
        sum += input.extract(i);
    }
    sum
}
