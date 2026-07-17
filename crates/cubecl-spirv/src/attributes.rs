use cubecl_core::{e2m1, e2m1x2, e4m3, e5m2, tf32, ue8m0};
use cubecl_ir::{
    ElemType, FloatKind, attributes::IndexAttr, interfaces::ScalarType, prelude::*, try_cast_ty,
    verify_attr_succ,
};

use half::{bf16, f16};
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{attr_interfaces::TypedAttrInterface, attributes::IntegerAttr},
    utils::{
        apfloat::double_to_f64,
        apint::{APInt, bw},
    },
};
use pliron_spirv::attrs::FloatAttr;

use crate::types::ty_to_spirv_dialect;

pub fn attr_to_spirv_dialect(ctx: &Context, attr: &AttrObj) -> AttrObj {
    if let Some(to_spirv_dialect) = attr_cast::<dyn ToSpirvDialectAttr>(&**attr) {
        to_spirv_dialect.to_spirv_dialect(ctx)
    } else {
        attr.clone()
    }
}

#[attr_interface]
pub trait ToSpirvDialectAttr {
    verify_attr_succ!();
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj;
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for IndexAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let value = self.0;
        let width = bw(ctx.address_type().size_bits());
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be integer");
        IntegerAttr::new(ty, APInt::from_usize(value, width)).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for IntegerAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let ty = ty_to_spirv_dialect(ctx, self.get_type());
        IntegerAttr::new(TypedHandle::from_handle(ty, ctx).unwrap(), self.value()).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for cubecl_ir::attributes::FloatAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let value = float_bits(ctx, double_to_f64(self.val), self.get_type(ctx));
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be float");
        FloatAttr::new(value, ty).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for cubecl_ir::attributes::BoolAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let value = if self.0 { 1 } else { 0 };
        let value = APInt::from_u8(value, bw(1));
        IntegerAttr::new(TypedHandle::from_handle(ty, ctx).unwrap(), value).into()
    }
}

fn float_bits(ctx: &Context, value: f64, ty: TypeHandle) -> u64 {
    let elem_ty = try_cast_ty!(ty.deref(ctx), ctx, dyn ScalarType).elem_type(ctx);
    let ElemType::Float(kind) = elem_ty else {
        panic!("Should be float")
    };
    match kind {
        FloatKind::E2M1 => e2m1::from_f64(value).to_bits() as u64,
        FloatKind::E2M1x2 => {
            e2m1x2::from_f32_slice(&[value as f32, value as f32])[0].to_bits() as u64
        }
        FloatKind::E2M3 | FloatKind::E3M2 => panic!("Unsupported constant type"),
        FloatKind::E4M3 => e4m3::from_f64(value).to_bits() as u64,
        FloatKind::E5M2 => e5m2::from_f64(value).to_bits() as u64,
        FloatKind::UE8M0 => ue8m0::from_f64(value).to_bits() as u64,
        FloatKind::F16 => f16::from_f64(value).to_bits() as u64,
        FloatKind::BF16 => bf16::from_f64(value).to_bits() as u64,
        FloatKind::Flex32 => (value as f32).to_bits() as u64,
        FloatKind::F32 => (value as f32).to_bits() as u64,
        FloatKind::TF32 => tf32::from_f64(value).to_bits() as u64,
        FloatKind::F64 => value.to_bits(),
    }
}
