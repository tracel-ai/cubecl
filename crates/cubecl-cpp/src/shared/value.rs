use cubecl_core::ir::{
    attributes::{BoolAttr, ComplexAttr, FloatAttr, IndexAttr, ZeroAttr},
    types::{barrier::BarrierTokenType, scalar::Complex32Type},
    verify_attr_succ,
};
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{attributes::IntegerAttr, ops::ConstantOp},
    common_traits::Named,
    context::Context,
    derive::{attr_interface, attr_interface_impl},
    identifier::Identifier,
    r#type::{TypeHandle, Typed},
    value::Value,
};

use crate::shared::{
    shared_op_with_out,
    ty::{TypeExtCPP, TypedExtCPP},
};

pub trait CppValue {
    fn name(&self, ctx: &Context) -> Identifier;
    fn fmt_left(&self, ctx: &Context) -> String;
}

impl CppValue for Value {
    fn name(&self, ctx: &Context) -> Identifier {
        self.unique_name(ctx)
    }

    fn fmt_left(&self, ctx: &Context) -> String {
        let ty = self.get_type(ctx).deref(ctx);
        let name = self.name(ctx);
        // C++ has weird semantics so this needs to be mutable for use with `std::move`.
        // `std::move` preserves constness for the moved value, and the API requires
        // a non-const `BarrierToken&&`.
        if ty.is::<BarrierTokenType>() {
            format!("{} {}", ty.to_cpp(ctx), name)
        } else {
            format!("{} const {}", ty.to_cpp(ctx), name)
        }
    }
}

#[attr_interface]
pub trait CppConstantAttr {
    verify_attr_succ!();
    fn as_f64(&self, ctx: &Context) -> f64;
    fn to_cpp(&self, ctx: &Context) -> String;
}

#[attr_interface_impl]
impl CppConstantAttr for IndexAttr {
    fn as_f64(&self, _ctx: &Context) -> f64 {
        self.0 as f64
    }
    fn to_cpp(&self, _ctx: &Context) -> String {
        format!("{}", self.0)
    }
}

#[attr_interface_impl]
impl CppConstantAttr for IntegerAttr {
    fn as_f64(&self, _ctx: &Context) -> f64 {
        self.value().to_i128() as f64
    }
    fn to_cpp(&self, ctx: &Context) -> String {
        let is_signed = self.get_type().deref(ctx).is_signed();
        self.value().to_string_decimal(is_signed)
    }
}

#[attr_interface_impl]
impl CppConstantAttr for FloatAttr {
    fn as_f64(&self, ctx: &Context) -> f64 {
        self.float_type(ctx).value_to_f64(self.val)
    }
    fn to_cpp(&self, ctx: &Context) -> String {
        // I would prefer to print the bits and use `bit_cast` but that's not well-supported. Keep
        // an eye on this to make sure it doesn't cause issues.
        self.float_type(ctx).value_to_string(self.val)
    }
}

#[attr_interface_impl]
impl CppConstantAttr for ComplexAttr {
    fn as_f64(&self, _ctx: &Context) -> f64 {
        f64::from_bits(self.re_bits)
    }
    fn to_cpp(&self, ctx: &Context) -> String {
        let re = f64::from_bits(self.re_bits);
        let im = f64::from_bits(self.im_bits);
        if self.ty.deref(ctx).is::<Complex32Type>() {
            format!("make_cuFloatComplex({:?}f, {:?}f)", re as f32, im as f32)
        } else {
            format!("make_cuDoubleComplex({re:?}, {im:?})")
        }
    }
}

#[attr_interface_impl]
impl CppConstantAttr for BoolAttr {
    fn as_f64(&self, _ctx: &Context) -> f64 {
        self.0 as u8 as f64
    }
    fn to_cpp(&self, _ctx: &Context) -> String {
        self.0.to_string()
    }
}

#[attr_interface_impl]
impl CppConstantAttr for ZeroAttr {
    fn as_f64(&self, _ctx: &Context) -> f64 {
        0.0
    }
    fn to_cpp(&self, _ctx: &Context) -> String {
        "{}".to_string()
    }
}

shared_op_with_out!(ConstantOp, |op, ctx| {
    format_const(ctx, op.get_value(ctx), op.get_result(ctx).get_type(ctx))
});

pub(crate) fn format_const(ctx: &Context, value: AttrObj, ty: TypeHandle) -> String {
    let const_attr = attr_cast::<dyn CppConstantAttr>(&*value).expect("Should be constant attr");
    if let Some(attr) = value.downcast_ref::<FloatAttr>() {
        let val = attr.float_type(ctx).value_to_f64(attr.val);
        // minifloats are represented as raw bits, so use special handling
        if ty.is_fp8_fp6_fp4(ctx) {
            format!("{}", attr.val.to_bits())
        } else if val.is_nan() {
            "(0.0f/0.0f)".into()
        } else if val.is_infinite() && val.is_sign_positive() {
            "(1.0f/0.0f)".into()
        } else if val.is_infinite() {
            "(-1.0f/0.0f)".into()
        } else {
            attr.to_cpp(ctx)
        }
    } else {
        const_attr.to_cpp(ctx)
    }
}
