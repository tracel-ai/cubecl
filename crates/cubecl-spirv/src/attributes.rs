use core::num::NonZeroUsize;

use cubecl_ir::{
    ContextExt,
    attributes::{IndexAttr, IntAttr, UIntAttr},
    verify_attr_succ,
};
use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{BoolAttr, IntegerAttr},
    },
    context::Context,
    derive::{attr_interface, attr_interface_impl},
    r#type::TypedHandle,
    utils::{apfloat::double_to_f64, apint::APInt},
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
        let width = NonZeroUsize::new(ctx.address_type().size_bits()).unwrap();
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be integer");
        IntegerAttr::new(ty, APInt::from_usize(value, width)).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for UIntAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let value = self.val;
        let width = NonZeroUsize::new(self.ty.deref(ctx).width).unwrap();
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be integer");
        IntegerAttr::new(ty, APInt::from_u64(value, width)).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for IntAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let value = self.val;
        let width = NonZeroUsize::new(self.ty.deref(ctx).width).unwrap();
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be integer");
        IntegerAttr::new(ty, APInt::from_i64(value, width)).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for cubecl_ir::attributes::FloatAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let value = double_to_f64(self.val);
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be float");
        FloatAttr::new(value.to_bits(), ty).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for cubecl_ir::attributes::BoolAttr {
    fn to_spirv_dialect(&self, _ctx: &Context) -> AttrObj {
        BoolAttr::new(self.0).into()
    }
}
