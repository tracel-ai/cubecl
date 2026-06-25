use cubecl_core::{
    self as cubecl, define_scalar, define_size,
    ir::{
        dialect::{
            bitwise::*,
            cmp::*,
            general::{BoolAndOp, BoolOrOp},
            math::*,
            memory::{CopyOp, IndexOp},
        },
        interfaces::TypedExt,
        prelude::*,
        types::ArrayType,
    },
    prelude::*,
};
use itertools::Itertools;

use crate::{
    cuda::packed_ops::packable,
    shared::{
        CppValue,
        convert::{no_half, promotes_int},
        shared_op, shared_op_with_out,
        ty::{PointerType, TypeExtCPP, TypedExtCPP},
        unroll::unrolling,
    },
    target::{CtxTarget, Hip},
};

macro_rules! operator {
    ($name:ident, $op:expr) => {
        shared_op_with_out!($name, |op, ctx| {
            let lhs = op.lhs(ctx).name(ctx);
            let rhs = op.rhs(ctx).name(ctx);
            format!("{lhs} {} {rhs}", $op)
        });
        unrolling!($name);
        promotes_int!($name);
    };
}

operator!(AddOp, "+");
operator!(SubOp, "-");
operator!(DivOp, "/");
operator!(MulOp, "*");
operator!(EqualOp, "==");
operator!(NotEqualOp, "!=");
operator!(LessThanOp, "<");
operator!(LessThanOrEqualOp, "<=");
operator!(GreaterThanOp, ">");
operator!(GreaterThanOrEqualOp, ">=");
operator!(ShiftLeftOp, "<<");
operator!(ShiftRightOp, ">>");
operator!(BitwiseOrOp, "|");
operator!(BitwiseAndOp, "&");
operator!(BitwiseXorOp, "^");
operator!(BoolOrOp, "||");
operator!(BoolAndOp, "&&");

shared_op_with_out!(RemOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    let out = op.get_result(ctx);
    if out.is_float32(ctx) | out.is_float64(ctx) {
        format!("fmod({lhs}, {rhs})")
    } else {
        format!("{lhs} % {rhs}")
    }
});
unrolling!(RemOp);
no_half!(RemOp);
promotes_int!(RemOp);

shared_op_with_out!(ModFloorOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    let out = op.get_result(ctx);
    let out_elem = out.get_type(ctx).to_cpp(ctx);
    let prefix = ctx.target().ty_prefix(ctx, out);
    let floor = format!("{prefix}floor");
    if out.is_int(ctx) {
        format!("{lhs} - {rhs} * ({out_elem}){floor}((float){lhs} / (float){rhs})")
    } else {
        format!("{lhs} - {rhs} * {floor}({lhs} / {rhs})")
    }
});
unrolling!(ModFloorOp);
packable!(ModFloorOp);

// pub struct FastDiv;

// impl<D: Dialect> Binary<D> for FastDiv {
//     fn format_scalar<Lhs: Display, Rhs: Display>(
//         f: &mut std::fmt::Formatter<'_>,
//         lhs: Lhs,
//         rhs: Rhs,
//         _out_item: Item<D>,
//     ) -> std::fmt::Result {
//         // f32 only
//         write!(f, "__fdividef({lhs}, {rhs})")
//     }
// }

shared_op_with_out!(MulHiOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    match (lhs.size(ctx), lhs.is_int(ctx)) {
        (4, true) => format!("__mulhi({}, {rhs})", lhs.name(ctx)),
        (4, false) => format!("__umulhi({}, {rhs})", lhs.name(ctx)),
        (8, true) => format!("__mul64hi({}, {rhs})", lhs.name(ctx)),
        (8, false) => format!("__umul64hi({}, {rhs})", lhs.name(ctx)),
        _ => unreachable!("HiMul only supports 32 and 64 bit ints"),
    }
});
unrolling!(MulHiOp);

macro_rules! lower_binop {
    ($ty: ty, $name: ident, $pred: expr) => {
        $crate::shared::binary::lower_target_binop!($ty, $name, $crate::target::Shared, $pred);
    };
    ($ty: ty, $name: ident) => {
        $crate::shared::binary::lower_binop!($ty, $name, |_, _| true);
    };
}
pub(crate) use lower_binop;

macro_rules! lower_target_binop {
    ($ty: ty, $name: ident, $target: ty, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::shared::lowering::LowerOp<$target> for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                $crate::shared::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                use cubecl_core::ir::dialect::OperationPtrExt;
                define_scalar!(T);
                define_size!(S);
                let lhs = self.get_operation().operand(scope.ctx(), 0);
                let rhs = self.get_operation().operand(scope.ctx(), 1);
                scope.register_value_type::<T, S>(rhs);
                vec![$name::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)]
            }
        }
    };
    ($ty: ty, $name: ident, $target: ty) => {
        lower_target_binop!($ty, $name, $target, |_, _| true);
    };
}
pub(crate) use lower_target_binop;

#[cube]
fn min_bf16<T: Numeric, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
    let lhs = Vector::<f32, N>::cast_from(lhs);
    let rhs = Vector::<f32, N>::cast_from(rhs);
    Vector::cast_from(lhs.min(rhs))
}

#[cube]
fn max_bf16<T: Numeric, N: Size>(lhs: Vector<T, N>, rhs: Vector<T, N>) -> Vector<T, N> {
    let lhs = Vector::<f32, N>::cast_from(lhs);
    let rhs = Vector::<f32, N>::cast_from(rhs);
    Vector::cast_from(lhs.min(rhs))
}

lower_target_binop!(MinOp, min_bf16, Hip, |op, ctx| {
    op.lhs(ctx).is_bfloat16(ctx)
});
lower_target_binop!(MaxOp, max_bf16, Hip, |op, ctx| {
    op.lhs(ctx).is_bfloat16(ctx)
});

shared_op_with_out!(MinOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    if lhs.is_half(ctx) {
        format!("__hmin({}, {rhs})", lhs.name(ctx))
    } else if lhs.is_half2(ctx) {
        format!("__hmin2({}, {rhs})", lhs.name(ctx))
    } else {
        format!("min({}, {rhs})", lhs.name(ctx))
    }
});
unrolling!(MinOp);
packable!(MinOp);
promotes_int!(MinOp);

shared_op_with_out!(MaxOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    if lhs.is_half(ctx) {
        format!("__hmax({}, {rhs})", lhs.name(ctx))
    } else if lhs.is_half2(ctx) {
        format!("__hmax2({}, {rhs})", lhs.name(ctx))
    } else {
        format!("max({}, {rhs})", lhs.name(ctx))
    }
});
unrolling!(MaxOp);
packable!(MaxOp);
promotes_int!(MaxOp);

shared_op_with_out!(ClampOp, |op, ctx| {
    let input = op.input(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    if input.is_half(ctx) {
        format!("__hmax(__hmin({}, {max}), {min})", input.name(ctx))
    } else if input.is_half2(ctx) {
        format!("__hmax2(__hmin2({}, {max}), {min})", input.name(ctx))
    } else {
        format!("max(min({}, {max}), {min})", input.name(ctx))
    }
});
unrolling!(ClampOp);
packable!(ClampOp);
promotes_int!(ClampOp);

shared_op_with_out!(PowfOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("pow({}, {rhs})", lhs.name(ctx))
});
unrolling!(PowfOp);
no_half!(PowfOp);

shared_op_with_out!(PowiOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("pow({}, {rhs})", lhs.name(ctx))
});
unrolling!(PowiOp);
no_half!(PowiOp);

// pub struct FastPowf;

// impl<D: Dialect> Binary<D> for FastPowf {
//     // Only executed for f32
//     fn format_scalar<Lhs: Display, Rhs: Display>(
//         f: &mut std::fmt::Formatter<'_>,
//         lhs: Lhs,
//         rhs: Rhs,
//         _item: Item<D>,
//     ) -> std::fmt::Result {
//         write!(f, "__powf({lhs}, {rhs})")
//     }
// }

shared_op_with_out!(ArcTan2Op, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("atan2({lhs}, {rhs})")
});
unrolling!(ArcTan2Op);
no_half!(ArcTan2Op);

shared_op_with_out!(HypotOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("hypot({}, {rhs})", lhs.name(ctx))
});
unrolling!(HypotOp);
no_half!(HypotOp);

shared_op_with_out!(RhypotOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    if lhs.is_float32(ctx) {
        format!("rhypotf({}, {rhs})", lhs.name(ctx))
    } else {
        format!("rhypot({}, {rhs})", lhs.name(ctx))
    }
});
unrolling!(RhypotOp);
no_half!(RhypotOp);

shared_op_with_out!(IndexOp, |op, ctx| {
    format!("&{}", fmt_index(ctx, op.base(ctx), op.index(ctx)))
});

pub fn fmt_index(ctx: &Context, list: Value, index: Value) -> String {
    let list_ty = list.get_type(ctx).deref(ctx);
    let list = list.name(ctx);
    let index = index.name(ctx);
    // Array nested in pointer, deref first
    if let Some(PointerType { inner, .. }) = list_ty.downcast_ref()
        && inner.deref(ctx).is::<ArrayType>()
    {
        format!("(*{list})[{index}]")
    } else {
        format!("{list}[{index}]")
    }
}

shared_op!(CopyOp, |op, ctx| {
    let source = op.source(ctx).name(ctx);
    let dest = op.destination(ctx).name(ctx);
    (0..op.len(ctx).0)
        .map(|i| format!("*({dest} + {i}) = *({source} + {i});\n"))
        .join("")
});
