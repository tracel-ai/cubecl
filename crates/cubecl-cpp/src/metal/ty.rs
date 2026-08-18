use cubecl_core::ir::{
    AddressSpace,
    prelude::*,
    types::{
        AtomicType, PointerType,
        scalar::{BFloat16Type, Float8E4M3Type, Float8E5M2Type, Float16Type},
    },
};

use crate::{
    shared::ty::{TypeExtCPP, TypeToCPP, UniformPointerType, ptr_constness},
    target::Metal,
};

macro_rules! metal_ty {
    ($ty: ty, $impl: expr) => {
        #[type_interface_impl]
        impl TypeToCPP<Metal> for $ty {
            fn to_cpp(&self, ctx: &Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use metal_ty;

metal_ty!(Float16Type, |_, _| "half".into());
metal_ty!(BFloat16Type, |_, _| "bfloat".into());
metal_ty!(Float8E4M3Type, |_, _| "uint8_t".into());
metal_ty!(Float8E5M2Type, |_, _| "uint8_t".into());

metal_ty!(PointerType, |ty, ctx| format!(
    "{} {} {}*",
    ptr_space(ty.address_space),
    ty.inner.to_cpp(ctx),
    ptr_constness(ctx, ty.address_space),
));
metal_ty!(UniformPointerType, |ty, ctx| format!(
    "constant {} const*",
    ty.inner.to_cpp(ctx)
));

pub fn ptr_space(addr_space: AddressSpace) -> &'static str {
    match addr_space {
        AddressSpace::Global(_) => "device",
        AddressSpace::Shared => "threadgroup",
        AddressSpace::Local => "thread",
    }
}

metal_ty!(AtomicType, |ty, ctx| format!(
    "atomic<{}>",
    ty.inner.to_cpp(ctx)
));
