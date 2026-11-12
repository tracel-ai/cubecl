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
use crate::target::{CtxTarget, Target};

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
    fn as_f64(&self, ctx: &Context) -> f64 {
        self.float_type(ctx).value_to_f64(self.re)
    }
    fn to_cpp(&self, ctx: &Context) -> String {
        let float_ty = self.float_type(ctx);
        let re = float_ty.value_to_string(self.re);
        let im = float_ty.value_to_string(self.im);
        if self.ty.deref(ctx).is::<Complex32Type>() {
            format!("make_cuFloatComplex({re}, {im})")
        } else {
            format!("make_cuDoubleComplex({re}, {im})")
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
    fn to_cpp(&self, ctx: &Context) -> String {
        // C has no empty initialiser for a scalar.
        match ctx.target() {
            Target::OpenCl => "{0}".to_string(),
            _ => "{}".to_string(),
        }
    }
}

shared_op_with_out!(ConstantOp, |op, ctx| {
    let attr = op.get_attr_builtin_constant_value(ctx).unwrap();
    format_const(ctx, &attr, op.get_result(ctx).get_type(ctx))
});

pub(crate) fn format_const(ctx: &Context, value: &AttrObj, ty: TypeHandle) -> String {
    let const_attr = attr_cast::<dyn CppConstantAttr>(&**value).expect("Should be constant attr");
    if let Some(attr) = value.downcast_ref::<FloatAttr>() {
        let val = attr.float_type(ctx).value_to_f64(attr.val);
        // minifloats are represented as raw bits, so use special handling
        if ty.is_fp8_fp6_fp4(ctx) {
            return format!("{}", attr.val.to_bits());
        }
        let literal = if val.is_nan() {
            "(0.0f/0.0f)".into()
        } else if val.is_infinite() && val.is_sign_positive() {
            "(1.0f/0.0f)".into()
        } else if val.is_infinite() {
            "(-1.0f/0.0f)".into()
        } else if val == 0.0 && val.is_sign_negative() {
            // Zero prints as the integer `-0`, which is `+0.0` once converted.
            "(-0.0f)".into()
        } else if ctx.target() == Target::OpenCl && !ty.is_f64(ctx) {
            // OpenCL C reads an unsuffixed literal as a `double`, which a device without fp64
            // can only narrow, with a warning.
            let literal = attr.to_cpp(ctx);
            match literal.contains(['.', 'e']) {
                true => format!("{literal}f"),
                false => format!("{literal}.0f"),
            }
        } else {
            attr.to_cpp(ctx)
        };
        // MSL's `bfloat` converts from `float` only explicitly.
        match ctx.target() == Target::Metal && ty.is_bf16(ctx) {
            true => format!("bfloat({literal})"),
            false => literal,
        }
    } else {
        const_attr.to_cpp(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shared::operation::OpToCPP, target::Shared};
    use cubecl_core::ir::{ConstantValue, ElemType, FloatKind};
    use pliron::{attribute::boxed_attr_cast, builtin::ops::ConstantOp};

    fn constant(target: Target, value: f64, kind: FloatKind) -> String {
        let mut ctx = Context::new();
        ctx.set_target(target);
        let attr = ConstantValue::Float(value).as_attribute(&ctx, ElemType::Float(kind));
        let op = ConstantOp::new(&mut ctx, boxed_attr_cast(attr).unwrap());
        let msl = OpToCPP::<Shared>::to_cpp(&op, &ctx);
        msl.split_once(" = ").unwrap().1.to_string()
    }

    /// A bf16 literal compiles in MSL only as an explicit conversion from `float`, NaN included.
    #[test]
    fn a_metal_bf16_literal_converts_explicitly() {
        assert_eq!(
            constant(Target::Metal, 0.5, FloatKind::BF16),
            "bfloat(0.5);\n"
        );
        assert_eq!(
            constant(Target::Metal, f64::NAN, FloatKind::BF16),
            "bfloat((0.0f/0.0f));\n"
        );
        assert_eq!(constant(Target::Metal, 0.5, FloatKind::F16), "0.5;\n");
    }

    /// `-0.0` as a float literal, on every target: the integer `-0` would be `+0.0`.
    #[test]
    fn negative_zero_keeps_its_sign() {
        for target in [Target::Cuda, Target::Hip, Target::Metal] {
            assert_eq!(constant(target, -0.0, FloatKind::F32), "(-0.0f);\n");
            assert_eq!(constant(target, 0.0, FloatKind::F32), "0;\n");
        }
        assert_eq!(
            constant(Target::Metal, -0.0, FloatKind::BF16),
            "bfloat((-0.0f));\n"
        );
    }

    /// Single precision, not a `double` the device narrows; a `double` stays one.
    #[test]
    fn an_opencl_literal_is_single_precision() {
        assert_eq!(constant(Target::OpenCl, 0.5, FloatKind::F32), "0.5f;\n");
        assert_eq!(constant(Target::OpenCl, 2.0, FloatKind::F32), "2.0f;\n");
        assert_eq!(constant(Target::OpenCl, 0.5, FloatKind::F16), "0.5f;\n");
        assert_eq!(constant(Target::OpenCl, 0.5, FloatKind::F64), "0.5;\n");
        assert_eq!(constant(Target::OpenCl, -0.0, FloatKind::F32), "(-0.0f);\n");
    }
}
