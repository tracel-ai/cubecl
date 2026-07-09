use cubecl_core::{
    self as cubecl,
    ir::{
        dialect::{
            atomic::AtomicLoadOp,
            base::ptr_value_ty,
            bitwise::{
                BitwiseNotOp, CountOnesOp, FindFirstSetOp, LeadingZerosBitsOp, ReverseBitsOp,
                TrailingZerosBitsOp,
            },
            general::{BoolNotOp, CastOp, ReinterpretCastOp},
            math::*,
            memory::{LoadOp, StoreOp},
            plane::UniformLoadOp,
        },
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};
use half::bf16;
use num_traits::{One, Zero};

use crate::{
    cuda::packed_ops::{PackableOp, packable},
    shared::{
        CppValue, OpToCPP,
        convert::{no_half, promotes_int},
        lowering::LowerOp,
        shared_op, shared_op_with_out,
        ty::{TypeExtCPP, TypedExtCPP},
        unroll::unrolling,
    },
    target::{CtxTarget, Shared, Target},
};

use core::f32::consts::PI;

pub trait FunctionFmt {
    fn base_function_name() -> &'static str;
    fn function_name(ctx: &Context, ty: impl Typed) -> String {
        let prefix = ctx.target().ty_prefix(ctx, ty);
        format!("{prefix}{}", Self::base_function_name())
    }
    fn format_unary(ctx: &Context, input: Value) -> String {
        let in_name = input.name(ctx);
        format!("{}({in_name})", Self::function_name(ctx, input))
    }
}

macro_rules! function {
    ($name:ident, $func:expr, $($flags: ident),*) => {
        impl FunctionFmt for $name {
            fn base_function_name() -> &'static str {
                $func
            }
        }

        #[op_interface_impl]
        impl OpToCPP<Shared> for $name {
            fn to_cpp(&self, ctx: &Context) -> String {
                format!(
                    "{} = {};",
                    self.get_result(ctx).fmt_left(ctx),
                    Self::format_unary(ctx, self.input(ctx))
                )
            }
        }
        unrolling!($name);
        $($flags!($name);)*
    };
}

function!(LogOp, "log", packable);
// function!(FastLog, "__logf", no_half);
function!(SinOp, "sin", packable);
function!(CosOp, "cos", packable);
function!(TanOp, "tan", no_half);
function!(TanhOp, "tanh", packable);
function!(SinhOp, "sinh", no_half);
function!(CoshOp, "cosh", no_half);
function!(ArcCosOp, "acos", no_half);
function!(ArcSinOp, "asin", no_half);
function!(ArcTanOp, "atan", no_half);
function!(ArcSinhOp, "asinh", no_half);
function!(ArcCoshOp, "acosh", no_half);
function!(ArcTanhOp, "atanh", no_half);
// function!(FastSinOp, "__sinf", false);
// function!(FastCosOp, "__cosf", false);
function!(SqrtOp, "sqrt", packable);
function!(RsqrtOp, "rsqrt", packable);
// function!(FastSqrt, "__fsqrt_rn", false);
// function!(FastInverseSqrt, "__frsqrt_rn", false);
function!(ExpOp, "exp", packable);
// function!(FastExp, "__expf", false);
function!(Expm1Op, "expm1", no_half);
function!(CeilOp, "ceil", packable);
function!(TruncOp, "trunc", packable);
function!(FloorOp, "floor", packable);
function!(RoundOp, "rint", packable);
// function!(FastRecip, "__frcp_rn", false);
// function!(FastTanhOp, "__tanhf", false);

function!(ErfOp, "erf", no_half);

shared_op_with_out!(AbsOp, |op, ctx| {
    let input = op.input(ctx);
    if input.is_half(ctx) {
        format!("__habs({})", input.name(ctx))
    } else if input.is_half2(ctx) {
        format!("__habs2({})", input.name(ctx))
    } else {
        format!("abs({})", input.name(ctx))
    }
});
unrolling!(AbsOp);
packable!(AbsOp);
promotes_int!(AbsOp);

shared_op_with_out!(NegOp, |op, ctx| format!("-{}", op.input(ctx).name(ctx)));
unrolling!(NegOp);
packable!(NegOp);
promotes_int!(NegOp);

shared_op_with_out!(BoolNotOp, |op, ctx| format!("!{}", op.input(ctx).name(ctx)));
unrolling!(BoolNotOp);

shared_op_with_out!(BitwiseNotOp, |op, ctx| format!(
    "~{}",
    op.input(ctx).name(ctx)
));
unrolling!(BitwiseNotOp);
promotes_int!(BitwiseNotOp);

// Handle bitcount stuff

shared_op_with_out!(CountOnesOp, |op, ctx| {
    let input = op.input(ctx);
    match input.size(ctx) {
        4 => format!("__popc({})", input.name(ctx)),
        8 => format!("__popcll({})", input.name(ctx)),
        _ => unreachable!("Unsupported size"),
    }
});
unrolling!(CountOnesOp);

shared_op_with_out!(ReverseBitsOp, |op, ctx| {
    let input = op.input(ctx);
    match input.size(ctx) {
        4 => format!("__brev({})", input.name(ctx)),
        8 => format!("__brevll({})", input.name(ctx)),
        _ => unreachable!("Unsupported size"),
    }
});
unrolling!(ReverseBitsOp);

shared_op_with_out!(LeadingZerosBitsOp, |op, ctx| {
    let input = op.input(ctx);
    match input.size(ctx) {
        4 => format!("__clz({})", input.name(ctx)),
        8 => format!("__clzll({})", input.name(ctx)),
        _ => unreachable!("Unsupported size"),
    }
});
unrolling!(LeadingZerosBitsOp);

shared_op_with_out!(FindFirstSetOp, |op, ctx| {
    let input = op.input(ctx);
    match input.size(ctx) {
        4 => format!("__ffs({})", input.name(ctx)),
        8 => format!("__ffsll({})", input.name(ctx)),
        _ => unreachable!("Unsupported size"),
    }
});
unrolling!(FindFirstSetOp);

shared_op_with_out!(CastOp, |op, ctx| {
    let input = op.input(ctx);
    let ty = input.get_type(ctx);
    format!("{}({})", ty.to_cpp(ctx), input.name(ctx))
});
unrolling!(CastOp);

// In and out packability might differ for cast. Also need to exempt bf16<->f16 because CUDA for some
// reason omits this cast in packed form, even though it allows packed casts to and from minifloats.
#[op_interface_impl]
impl PackableOp for CastOp {
    fn should_pack(&self, ctx: &Context) -> bool {
        let is_bf16_to_half = is_bf16(ctx, self.input(ctx)) && is_f16(ctx, self.get_result(ctx));
        let is_half_to_bf16 = is_f16(ctx, self.input(ctx)) && is_bf16(ctx, self.get_result(ctx));
        let can_pack_both = self.input(ctx).can_pack(ctx) && self.get_result(ctx).can_pack(ctx);
        !is_bf16_to_half && !is_half_to_bf16 && can_pack_both
    }
}

fn is_f16(ctx: &Context, val: Value) -> bool {
    val.try_get_scalar_ty(ctx)
        .is_some_and(|scalar| scalar.is_float16(ctx))
}

fn is_bf16(ctx: &Context, val: Value) -> bool {
    val.try_get_scalar_ty(ctx)
        .is_some_and(|scalar| scalar.is_bfloat16(ctx))
}

shared_op_with_out!(ReinterpretCastOp, |op, ctx| {
    let input = op.input(ctx);
    let ty = op.get_result(ctx).get_type(ctx).to_cpp(ctx);
    if input.is_ptr(ctx) {
        format!("reinterpret_cast<{ty}>({})", input.name(ctx))
    } else {
        format!("reinterpret_cast<const {ty}&>({})", input.name(ctx))
    }
});

shared_op_with_out!(LoadOp, |op, ctx| format!("*{}", op.ptr(ctx).name(ctx)));
shared_op!(StoreOp, |op, ctx| {
    format!("*{} = {};", op.ptr(ctx).name(ctx), op.value(ctx).name(ctx))
});

macro_rules! lower_unop {
    ($ty: ty, $name: ident, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::shared::lowering::LowerOp for $ty {
            fn should_lower(&self, ctx: &Context) -> bool {
                $crate::shared::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                define_scalar!(T);
                define_size!(S);
                let input = self.get_operand(scope.ctx());
                scope.register_value_type::<T, S>(input);
                vec![$name::expand::<T, S>(scope, input.into()).read_value(scope)]
            }
        }
    };
    ($ty: ty, $name: ident) => {
        lower_unop!($ty, $name, |_, _| true);
    };
}
pub(super) use lower_unop;

#[cube]
fn log1p<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    (input + Vector::new(T::new(1.0))).ln()
}

#[cube]
fn to_degrees<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    input * Vector::new(T::new(comptime!(180.0 / PI)))
}

#[cube]
fn to_radians<T: Float, N: Size>(input: Vector<T, N>) -> Vector<T, N> {
    input * Vector::new(T::new(comptime!(PI / 180.0)))
}

#[cube]
fn find_first_set<T: Int, N: Size>(input: Vector<T, N>) -> Vector<u32, N> {
    let bits = Vector::new(comptime!(T::size_bits() as u32));
    let out = bits - (input & (!input - Vector::one())).leading_zeros();
    select_many(input.equal(&Vector::zero()), Vector::zero(), out)
}

#[cube]
fn trailing_zeros<T: Int, N: Size>(input: Vector<T, N>) -> Vector<u32, N> {
    let bits = Vector::new(T::size_bits().comptime() as u32);
    let out = input.find_first_set() - Vector::one();
    select_many(input.equal(&Vector::zero()), bits, out)
}

#[cube]
fn cast_f16_bf16<T: Scalar, N: Size>(input: Vector<T, N>) -> Vector<bf16, N> {
    Vector::<bf16, N>::cast_from(Vector::<f32, N>::cast_from(input))
}

#[cube]
fn count_ones<T: Scalar, N: Size>(input: Vector<T, N>) -> Vector<u32, N> {
    Vector::<u32, N>::cast_from(Vector::<u32, N>::cast_from(input).count_ones())
}

lower_unop!(Log1pOp, log1p);
lower_unop!(DegreesOp, to_degrees);
lower_unop!(RadiansOp, to_radians);
lower_unop!(FindFirstSetOp, find_first_set, |_, ctx| {
    ctx.target() == Target::Metal
});
lower_unop!(TrailingZerosBitsOp, trailing_zeros, |_, ctx| {
    matches!(ctx.target(), Target::Cuda | Target::Hip)
});
lower_unop!(ErfOp, erf, |_, ctx| ctx.target() == Target::Metal);
lower_unop!(CastOp, cast_f16_bf16, |op, ctx| {
    op.input(ctx).is_float16(ctx)
        && op.get_result(ctx).is_bfloat16(ctx)
        && matches!(ctx.target(), Target::Cuda | Target::Hip)
});

// `isnan` / `isinf` are defined for cuda/hip/metal with same prefixes for half/bf16 on cuda/hip

fn elem_function_name(ctx: &Context, base_name: &'static str, ty: impl Typed) -> String {
    // Math functions prefix (no leading underscores)
    let prefix = ctx.target().ty_prefix(ctx, ty);
    if prefix.is_empty() {
        base_name.to_string()
    } else if prefix == "h" || prefix == "h2" {
        format!("__{prefix}{base_name}")
    } else {
        panic!("Unknown prefix '{prefix}'");
    }
}

shared_op_with_out!(IsNanOp, |op, ctx| {
    let input = op.input(ctx);
    let func = elem_function_name(ctx, "isnan", input);
    format!("{func}({})", input.name(ctx))
});
unrolling!(IsNanOp);

shared_op_with_out!(IsInfOp, |op, ctx| {
    let input = op.input(ctx);
    let func = elem_function_name(ctx, "isinf", input);
    format!("{func}({})", input.name(ctx))
});
unrolling!(IsInfOp);

#[op_interface_impl]
impl LowerOp for UniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ptr = self.ptr(scope.ctx());
        let val = if ptr_value_ty(scope.ctx(), &ptr).is_atomic(scope.ctx()) {
            let op = AtomicLoadOp::new(scope.ctx_mut(), ptr);
            scope.register_with_result(&op)
        } else {
            let op = LoadOp::new(scope.ctx_mut(), ptr);
            scope.register_with_result(&op)
        };
        vec![val]
    }
}
