use cubecl_core::{
    cmma::MatrixType,
    ir::{
        prelude::*,
        types::{
            PointerType,
            scalar::{BFloat16Type, Float16Type},
        },
    },
};

use crate::{
    shared::{
        signature::ty_includes,
        ty::{TypeExtCPP, TypeToCPP, UniformPointerType, ptr_constness},
    },
    target::Hip,
};

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

hip_ty!(PointerType, |ty, ctx| format!(
    "{} {}*",
    ty.inner.to_cpp(ctx),
    ptr_constness(ctx, ty.address_space),
));
hip_ty!(UniformPointerType, |ty, ctx| format!(
    "{} const*",
    ty.inner.to_cpp(ctx)
));

ty_includes!(Hip, [MatrixType] => "rocwmma/rocwmma.hpp");
// hiprtc's builtin header only declares the legacy `hip_bfloat16` struct;
// `__hip_bfloat16` (the CUDA-compatible type emitted above) needs the real
// header, exactly as the pre-pliron dialect included it whenever bf16 appeared.
// (`__half` needs no include: the builtin header covers it.)
ty_includes!(Hip, [BFloat16Type, crate::cuda::ty::BFloat16x2Type] => "hip/hip_bf16.h");
