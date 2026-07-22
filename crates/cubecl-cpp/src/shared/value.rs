use cubecl_core::{
    e2m1, e4m3, e5m2,
    ir::{
        attributes::{BoolAttr, FloatAttr, IndexAttr},
        types::{barrier::BarrierTokenType, scalar::*},
        verify_attr_succ,
    },
    ue8m0,
};
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{attributes::IntegerAttr, ops::ConstantOp},
    common_traits::Named,
    context::Context,
    derive::{attr_interface, attr_interface_impl},
    identifier::Identifier,
    r#type::{TypeHandle, Typed},
    utils::apfloat::{Float, double_to_f64},
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

macro_rules! const_attr {
    ($ty: ty, $val: expr) => {
        #[attr_interface_impl]
        impl CppConstantAttr for $ty {
            fn as_f64(&self, ctx: &Context) -> f64 {
                $crate::shared::closure_inference_hack::<Self, _>(self, ctx, $val) as f64
            }
            fn to_cpp(&self, ctx: &Context) -> String {
                let val = $crate::shared::closure_inference_hack::<Self, _>(self, ctx, $val);
                format!("{val}")
            }
        }
    };
}

const_attr!(IntegerAttr, |attr, _| attr.value().to_i128());
const_attr!(FloatAttr, |attr, _| double_to_f64(attr.val));
const_attr!(IndexAttr, |attr, _| attr.0);

#[attr_interface_impl]
impl CppConstantAttr for BoolAttr {
    fn as_f64(&self, _ctx: &Context) -> f64 {
        self.0 as u8 as f64
    }
    fn to_cpp(&self, _ctx: &Context) -> String {
        self.0.to_string()
    }
}

shared_op_with_out!(ConstantOp, |op, ctx| {
    format_const(ctx, op.get_value(ctx), op.get_result(ctx).get_type(ctx))
});

pub(crate) fn format_const(ctx: &Context, value: AttrObj, ty: TypeHandle) -> String {
    let const_attr = attr_cast::<dyn CppConstantAttr>(&*value).expect("Should be constant attr");
    // minifloats are represented as raw bits, so use special handling
    if ty.deref(ctx).is::<Float4E2M1Type>() {
        format!("{}", e2m1::from_f64(const_attr.as_f64(ctx)).to_bits())
    } else if ty.is_float6(ctx) {
        todo!("FP6 constants are not yet supported")
    } else if ty.deref(ctx).is::<Float8E4M3Type>() {
        format!("{}", e4m3::from_f64(const_attr.as_f64(ctx)).to_bits())
    } else if ty.deref(ctx).is::<Float8E5M2Type>() {
        format!("{}", e5m2::from_f64(const_attr.as_f64(ctx)).to_bits())
    } else if ty.deref(ctx).is::<Float8E8M0Type>() {
        format!("{}", ue8m0::from_f64(const_attr.as_f64(ctx)).to_bits())
    } else if let Some(attr) = value.downcast_ref::<FloatAttr>() {
        if attr.val.is_nan() {
            "(0.0f/0.0f)".into()
        } else if attr.val.is_pos_infinity() {
            "(1.0f/0.0f)".into()
        } else if attr.val.is_neg_infinity() {
            "(-1.0f/0.0f)".into()
        } else {
            attr.to_cpp(ctx)
        }
    } else {
        const_attr.to_cpp(ctx)
    }
}
