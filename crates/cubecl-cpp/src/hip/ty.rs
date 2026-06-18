use cubecl_core::ir::{
    prelude::*,
    types::scalar::{BFloat16Type, Float16Type},
};

use crate::{shared::ty::TypeToCPP, target::Hip};

macro_rules! hip_ty {
    ($ty: ty, $impl: expr) => {
        #[type_interface_impl]
        impl TypeToCPP<Hip> for $ty {
            fn to_cpp(&self, ctx: &Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use hip_ty;

hip_ty!(Float16Type, |_, _| "__half".into());
hip_ty!(BFloat16Type, |_, _| "__hip_bfloat16".into());
