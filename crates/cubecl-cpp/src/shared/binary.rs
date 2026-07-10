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
        types::{ArrayType, PointerType},
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
        ty::{TypeExtCPP, TypedExtCPP},
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

operator!(IAddOp, "+");
operator!(FAddOp, "+");
operator!(ISubOp, "-");
operator!(FSubOp, "-");
operator!(SDivOp, "/");
operator!(UDivOp, "/");
operator!(FDivOp, "/");
operator!(IMulOp, "*");
operator!(FMulOp, "*");
operator!(IEqualOp, "==");
operator!(FEqualOp, "==");
operator!(INotEqualOp, "!=");
operator!(FNotEqualOp, "!=");
operator!(SLessThanOp, "<");
operator!(ULessThanOp, "<");
operator!(FLessThanOp, "<");
operator!(SLessThanOrEqualOp, "<=");
operator!(ULessThanOrEqualOp, "<=");
operator!(FLessThanOrEqualOp, "<=");
operator!(SGreaterThanOp, ">");
operator!(UGreaterThanOp, ">");
operator!(FGreaterThanOp, ">");
operator!(SGreaterThanOrEqualOp, ">=");
operator!(UGreaterThanOrEqualOp, ">=");
operator!(FGreaterThanOrEqualOp, ">=");
operator!(ShiftLeftOp, "<<");
operator!(ShiftRightOp, ">>");
operator!(BitwiseOrOp, "|");
operator!(BitwiseAndOp, "&");
operator!(BitwiseXorOp, "^");
operator!(BoolOrOp, "||");
operator!(BoolAndOp, "&&");

shared_op_with_out!(SRemOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("{lhs} % {rhs}")
});
unrolling!(SRemOp);
promotes_int!(SRemOp);

shared_op_with_out!(URemOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("{lhs} % {rhs}")
});
unrolling!(URemOp);
promotes_int!(URemOp);

shared_op_with_out!(FRemOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("fmod({lhs}, {rhs})")
});
unrolling!(FRemOp);
no_half!(FRemOp);

shared_op_with_out!(SModFloorOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    let out_elem = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    format!("{lhs} - {rhs} * ({out_elem})floor((float){lhs} / (float){rhs})")
});
unrolling!(SModFloorOp);

shared_op_with_out!(FModFloorOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    let prefix = ctx.target().ty_prefix(ctx, op.get_result(ctx));
    let floor = format!("{prefix}floor");
    format!("{lhs} - {rhs} * {floor}({lhs} / {rhs})")
});
unrolling!(FModFloorOp);
packable!(FModFloorOp);

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

shared_op_with_out!(SMulHiOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    match lhs.size(ctx) {
        4 => format!("__mulhi({}, {rhs})", lhs.name(ctx)),
        8 => format!("__mul64hi({}, {rhs})", lhs.name(ctx)),
        _ => unreachable!("HiMul only supports 32 and 64 bit ints"),
    }
});
unrolling!(SMulHiOp);

shared_op_with_out!(UMulHiOp, |op, ctx| {
    let lhs = op.lhs(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    match lhs.size(ctx) {
        4 => format!("__umulhi({}, {rhs})", lhs.name(ctx)),
        8 => format!("__umul64hi({}, {rhs})", lhs.name(ctx)),
        _ => unreachable!("HiMul only supports 32 and 64 bit ints"),
    }
});
unrolling!(UMulHiOp);

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

lower_target_binop!(FMinOp, min_bf16, Hip, |op, ctx| {
    op.lhs(ctx).is_bfloat16(ctx)
});
lower_target_binop!(FMaxOp, max_bf16, Hip, |op, ctx| {
    op.lhs(ctx).is_bfloat16(ctx)
});

shared_op_with_out!(SMinOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("min({lhs}, {rhs})")
});
unrolling!(SMinOp);
promotes_int!(SMinOp);

shared_op_with_out!(UMinOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("min({lhs}, {rhs})")
});
unrolling!(UMinOp);
promotes_int!(UMinOp);

shared_op_with_out!(FMinOp, |op, ctx| {
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
unrolling!(FMinOp);
packable!(FMinOp);

shared_op_with_out!(SMaxOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("max({lhs}, {rhs})")
});
unrolling!(SMaxOp);
promotes_int!(SMaxOp);

shared_op_with_out!(UMaxOp, |op, ctx| {
    let lhs = op.lhs(ctx).name(ctx);
    let rhs = op.rhs(ctx).name(ctx);
    format!("max({lhs}, {rhs})")
});
unrolling!(UMaxOp);
promotes_int!(UMaxOp);

shared_op_with_out!(FMaxOp, |op, ctx| {
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
unrolling!(FMaxOp);
packable!(FMaxOp);

shared_op_with_out!(SClampOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("max(min({input}, {max}), {min})")
});
unrolling!(SClampOp);
promotes_int!(SClampOp);

shared_op_with_out!(UClampOp, |op, ctx| {
    let input = op.input(ctx).name(ctx);
    let min = op.min(ctx).name(ctx);
    let max = op.max(ctx).name(ctx);
    format!("max(min({input}, {max}), {min})")
});
unrolling!(UClampOp);
promotes_int!(UClampOp);

shared_op_with_out!(FClampOp, |op, ctx| {
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
unrolling!(FClampOp);
packable!(FClampOp);

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
