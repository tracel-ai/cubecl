use cubecl_ir::{attributes::IndexAttr, prelude::*, verify_attr_succ};

use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{attr_interfaces::TypedAttrInterface, attributes::IntegerAttr},
    utils::apint::{APInt, bw},
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
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be float");
        FloatAttr::new(self.val.to_bits() as u64, ty).into()
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
