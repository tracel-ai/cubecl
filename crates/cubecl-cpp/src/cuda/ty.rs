use cubecl_core::{
    cmma::MatrixType,
    ir::{
        aligned,
        pliron::context::Context,
        scalar, sized,
        types::{
            PointerType,
            barrier::{BarrierLevel, BarrierTokenType, BarrierType},
            cuda::TensorMapType,
            scalar::*,
        },
    },
};
use pliron::derive::{format, pliron_type, type_interface_impl};

use crate::{
    shared::{
        signature::{RequiresIncludesType, ty_includes},
        ty::{TypeExtCPP, UniformPointerType, ptr_constness},
    },
    target::Cuda,
};

macro_rules! cuda_ty {
    ($ty: ty, $impl: expr) => {
        #[type_interface_impl]
        impl crate::shared::ty::TypeToCPP<crate::target::Cuda> for $ty {
            fn to_cpp(&self, ctx: &Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use cuda_ty;

cuda_ty!(TensorMapType, |_, _| "CUtensorMap".into());
cuda_ty!(BarrierType, |ty, _| match ty.0 {
    BarrierLevel::Unit => "cuda::barrier<cuda::thread_scope_thread>".into(),
    BarrierLevel::Cube => "cuda::barrier<cuda::thread_scope_block>".into(),
});
cuda_ty!(BarrierTokenType, |ty, ctx| {
    format!("{}::arrival_token", ty.0.to_cpp(ctx))
});

#[type_interface_impl]
impl RequiresIncludesType<Cuda> for BarrierType {
    fn includes(&self, _ctx: &Context) -> Vec<String> {
        vec![
            "cuda/barrier".into(),
            "cooperative_groups.h".into(),
            "cooperative_groups/memcpy_async.h".into(),
        ]
    }
}

cuda_ty!(PointerType, |ty, ctx| format!(
    "{} {}*",
    ty.inner.to_cpp(ctx),
    ptr_constness(ctx, ty.address_space),
));
cuda_ty!(UniformPointerType, |ty, ctx| format!(
    "{} const*",
    ty.inner.to_cpp(ctx)
));

#[pliron_type(
    name = "cpp.f16x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(new, Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct Float16x2Type;
sized!(Float16x2Type, size_of::<u32>());
aligned!(Float16x2Type, align_of::<u32>());
scalar!(Float16x2Type);

#[pliron_type(
    name = "cpp.bf16x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(new, Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct BFloat16x2Type;
sized!(BFloat16x2Type, size_of::<u32>());
aligned!(BFloat16x2Type, align_of::<u32>());
scalar!(BFloat16x2Type);

#[pliron_type(
    name = "cuda.ue8m0x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Float8E8M0x2Type;
sized!(Float8E8M0x2Type, size_of::<u16>());
aligned!(Float8E8M0x2Type, align_of::<u16>());
scalar!(Float8E8M0x2Type);

#[pliron_type(
    name = "cuda.e4m3x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Float8E4M3x2Type;
sized!(Float8E4M3x2Type, size_of::<u16>());
aligned!(Float8E4M3x2Type, align_of::<u16>());
scalar!(Float8E4M3x2Type);

#[pliron_type(
    name = "cuda.e5m2x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Float8E5M2x2Type;
sized!(Float8E5M2x2Type, size_of::<u16>());
aligned!(Float8E5M2x2Type, align_of::<u16>());
scalar!(Float8E5M2x2Type);

#[pliron_type(
    name = "cuda.e3m2x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Float6E3M2x2Type;
sized!(Float6E3M2x2Type, size_of::<u16>());
aligned!(Float6E3M2x2Type, align_of::<u16>());
scalar!(Float6E3M2x2Type);

#[pliron_type(
    name = "cuda.e2m3x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Float6E2M3x2Type;
sized!(Float6E2M3x2Type, size_of::<u16>());
aligned!(Float6E2M3x2Type, align_of::<u16>());
scalar!(Float6E2M3x2Type);

ty_includes!(Cuda, [MatrixType, TFloat32Type] => "mma.h");
ty_includes!(Cuda, [Float16Type, Float16x2Type] => "cuda_fp16.h");
ty_includes!(Cuda, [BFloat16Type, BFloat16x2Type] => "cuda_bf16.h");
ty_includes!(Cuda, [Float8E4M3Type, Float8E5M2Type, Float8E8M0Type] => "cuda_fp8.h");
ty_includes!(Cuda, [Float8E4M3x2Type, Float8E5M2x2Type, Float8E8M0x2Type] => "cuda_fp8.h");
ty_includes!(Cuda, [Float6E3M2Type, Float6E2M3Type, Float6E3M2x2Type, Float6E2M3x2Type] => "cuda_fp6.h");
ty_includes!(Cuda, [Float4E2M1Type, Float4E2M1x2Type] => "cuda_fp4.h");

cuda_ty!(TFloat32Type, |_, _| "float".into());

cuda_ty!(Float16x2Type, |_, _| "__half2".into());
cuda_ty!(BFloat16x2Type, |_, _| "__nv_bfloat162".into());

cuda_ty!(Float8E4M3x2Type, |_, _| "__nv_fp8x2_storage_t".into());
cuda_ty!(Float8E5M2x2Type, |_, _| "__nv_fp8x2_storage_t".into());
cuda_ty!(Float8E8M0x2Type, |_, _| "__nv_fp8x2_storage_t".into());

cuda_ty!(Float6E3M2x2Type, |_, _| "__nv_fp6x2_storage_t".into());
cuda_ty!(Float6E2M3x2Type, |_, _| "__nv_fp6x2_storage_t".into());

cuda_ty!(Float4E2M1x2Type, |_, _| "__nv_fp4x2_storage_t".into());

cuda_ty!(Float16Type, |_, _| "__half".into());
cuda_ty!(BFloat16Type, |_, _| "__nv_bfloat16".into());

cuda_ty!(Float8E4M3Type, |_, _| "__nv_fp8_storage_t".into());
cuda_ty!(Float8E5M2Type, |_, _| "__nv_fp8_storage_t".into());
cuda_ty!(Float8E8M0Type, |_, _| "__nv_fp8_storage_t".into());

cuda_ty!(Float6E3M2Type, |_, _| "__nv_fp6_storage_t".into());
cuda_ty!(Float6E2M3Type, |_, _| "__nv_fp6_storage_t".into());

cuda_ty!(Float4E2M1Type, |_, _| "__nv_fp4_storage_t".into());
