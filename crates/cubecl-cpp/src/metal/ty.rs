use cubecl_core::ir::{
    prelude::*,
    types::scalar::{BFloat16Type, Float16Type},
};

use crate::{shared::ty::TypeToCPP, target::Metal};

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
