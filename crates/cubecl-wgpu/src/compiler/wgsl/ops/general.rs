use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{
    attributes::{BoolAttr, FloatAttr, IndexAttr},
    dialect::general::*,
    interfaces::TypedExt,
    prelude::*,
    types::{
        VectorType,
        scalar::{Float32Type, IndexType},
    },
};
use pliron::{
    attribute::{Attribute, attr_cast},
    builtin::{attributes::IntegerAttr, ops::ConstantOp, types::IntegerType},
};

use crate::compiler::wgsl::{
    lower::lower_binop,
    to_wgsl::{AttrToWgsl, TypeExtWgsl, wgsl_op, wgsl_op_with_out},
    value::WgslValue,
};

wgsl_op_with_out!(CopyOp; |op, ctx| op.value(ctx).name(ctx).into());

wgsl_op_with_out!(BoolAndOp; |op, ctx| {
    format!("{} && {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(BoolOrOp; |op, ctx| {
    format!("{} || {}", op.lhs(ctx).name(ctx), op.rhs(ctx).name(ctx))
});
wgsl_op_with_out!(BoolNotOp; |op, ctx| {
    format!("!{}", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(CastOp; |op, ctx| {
    fmt_cast_to(ctx, op.input(ctx), op.result_type(ctx))
});
wgsl_op_with_out!(ReinterpretCastOp; |op, ctx| {
    let ty = op.result_type(ctx).to_wgsl(ctx);
    format!("bitcast<{ty}>({})", op.input(ctx).name(ctx))
});

wgsl_op_with_out!(SelectOp; |op, ctx| {
    let t = op.true_value(ctx).name(ctx);
    let f = op.false_value(ctx).name(ctx);
    format!("select({f}, {t}, {})", op.condition(ctx).name(ctx))
});

wgsl_op!(FreeOp, |_, _| String::new());
wgsl_op!(PrintfOp, |_, _| String::new()); // Unsupported
wgsl_op!(CommentOp, |op, ctx| {
    let comment = op.comment(ctx).as_str().to_owned();
    if comment.contains('\n') {
        format!("/* {comment} */\n")
    } else {
        format!("// {comment}\n")
    }
});

wgsl_op_with_out!(ConstantOp; |op, ctx| attr_to_wgsl(ctx, &*op.get_value(ctx)));

pub fn attr_to_wgsl(ctx: &Context, attr: &dyn Attribute) -> String {
    let Some(attr) = attr_cast::<dyn AttrToWgsl>(attr) else {
        panic!("Constant value must implement `AttrToWgsl`")
    };
    attr.to_wgsl(ctx)
}

#[attr_interface_impl]
impl AttrToWgsl for IndexAttr {
    fn to_wgsl(&self, ctx: &Context) -> String {
        format!("{}({})", IndexType::get(ctx).to_wgsl(ctx), self.0)
    }
}

#[attr_interface_impl]
impl AttrToWgsl for IntegerAttr {
    fn to_wgsl(&self, ctx: &Context) -> String {
        let ty = self.get_type().to_handle();
        let val = self.value();
        // naga can't seem to parse literals > i64::MAX or i64::MIN atm.
        // Work around this by emitting instructions to construct these literals.
        if ty.is_unsigned_int(ctx) && val.to_u64() > i64::MAX as u64 {
            let as_i64 = val.to_u64() as i64;
            if as_i64 == i64::MIN {
                "bitcast<u64>(i64(-9223372036854775807) - 1)".into()
            } else {
                format!("bitcast<u64>(i64({as_i64}))")
            }
        } else if ty.is_signed_int(ctx) && val.to_i64() == i64::MIN {
            "(i64(-9223372036854775807) - 1)".into()
        } else {
            let val = val.to_string_decimal(ty.is_signed_int(ctx));
            format!("{}({val})", ty.to_wgsl(ctx))
        }
    }
}

#[attr_interface_impl]
impl AttrToWgsl for FloatAttr {
    fn to_wgsl(&self, ctx: &Context) -> String {
        format!("{}({})", self.ty.to_wgsl(ctx), self.val)
    }
}

#[attr_interface_impl]
impl AttrToWgsl for BoolAttr {
    fn to_wgsl(&self, _ctx: &Context) -> String {
        format!("{}", self.0)
    }
}

pub fn fmt_cast_to(ctx: &Context, value: Value, to: TypeHandle) -> String {
    let from = value.get_type(ctx);
    let from_elem = from.scalar_ty(ctx);
    let to_elem = to.scalar_ty(ctx);

    // Naga u64/i64 has weird limitations. We work around by first bitcasting to a 64-bit
    // type matching the target's signedness, then casting to the 32-bit target.
    let is_64bit = from_elem.is_int_of_width(ctx, 64);
    let is_32bit_target = to_elem.is_int_of_width(ctx, 32);
    if is_64bit && is_32bit_target {
        // Choose bitcast type based on target signedness (u32 -> u64, i32 -> i64)
        let to_elem = TypedHandle::<IntegerType>::from_handle(to_elem, ctx).unwrap();
        let bitcast_elem = IntegerType::get(ctx, 64, to_elem.deref(ctx).signedness());
        let bitcast_ty = same_vec_with_elem(ctx, from, bitcast_elem.into());

        let to_ty = to.to_wgsl(ctx);
        let bitcast_ty = bitcast_ty.to_wgsl(ctx);
        return format!("{to_ty}(bitcast<{bitcast_ty}>({}))", value.name(ctx));
    }

    // WGSL doesn't support direct bool to f16 casts, can go through f32 first.
    if from_elem.is_bool(ctx) && to_elem.is_float16(ctx) {
        let f32_ty = same_vec_with_elem(ctx, to, Float32Type::get(ctx).into()).to_wgsl(ctx);
        return format!("{}({f32_ty}({}))", to.to_wgsl(ctx), value.name(ctx));
    }

    format!("{}({})", to.to_wgsl(ctx), value.name(ctx))
}

fn same_vec_with_elem(ctx: &Context, maybe_vec: TypeHandle, elem: TypeHandle) -> TypeHandle {
    let vec = maybe_vec.vector_size(ctx);
    if vec > 1 {
        VectorType::get(ctx, elem, vec).to_handle()
    } else {
        elem
    }
}

lower_binop!(BoolAndOp, unroll_bool_and, |op, ctx| {
    op.result_type(ctx).vector_size(ctx) > 1
});
lower_binop!(BoolOrOp, unroll_bool_or, |op, ctx| {
    op.result_type(ctx).vector_size(ctx) > 1
});

#[cube]
#[allow(clippy::extra_unused_type_parameters)]
fn unroll_bool_and<T: Scalar, N: Size>(
    lhs: Vector<bool, N>,
    rhs: Vector<bool, N>,
) -> Vector<bool, N> {
    let mut out = Vector::empty();
    #[unroll]
    for i in 0..lhs.vector_size() {
        out.insert(i, lhs.extract(i) && rhs.extract(i));
    }
    out
}

#[cube]
#[allow(clippy::extra_unused_type_parameters)]
fn unroll_bool_or<T: Scalar, N: Size>(
    lhs: Vector<bool, N>,
    rhs: Vector<bool, N>,
) -> Vector<bool, N> {
    let mut out = Vector::empty();
    #[unroll]
    for i in 0..lhs.vector_size() {
        out.insert(i, lhs.extract(i) || rhs.extract(i));
    }
    out
}
