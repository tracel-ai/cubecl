use cubecl_core::ir::{
    dialect::synchronization::{SyncAsyncProxyOp, SyncOp, SyncScope},
    prelude::*,
};

use crate::{shared::signature::op_includes, target::Cuda};

macro_rules! cuda_op {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Cuda> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl)
            }
        }
    };
}
pub(super) use cuda_op;

macro_rules! cuda_op_with_out {
    ($ty: ty, $impl: expr) => {
        #[pliron::derive::op_interface_impl]
        impl $crate::shared::operation::OpToCPP<$crate::target::Cuda> for $ty {
            fn to_cpp(&self, ctx: &pliron::context::Context) -> String {
                use cubecl_core::ir::prelude::*;
                use $crate::shared::CppValue;
                let op = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $impl);
                let out = self.get_result(ctx).fmt_left(ctx);
                format!("{out} = {op};\n")
            }
        }
    };
}
pub(super) use cuda_op_with_out;

macro_rules! ptx_with_out {
    ($ty: ty, $ptx: expr, $pred: expr) => {
        #[op_interface_impl]
        impl $crate::shared::lowering::LowerOp<$crate::target::Cuda> for $ty {
            fn should_lower(&self, ctx: &pliron::context::Context) -> bool {
                $crate::shared::closure_inference_hack::<$ty, bool>(self, ctx, $pred)
            }
            fn lower(&self, scope: &cubecl_core::ir::Scope) -> Vec<pliron::value::Value> {
                use cubecl_core::ir::dialect::base::OperationPtrExt;
                use pliron::{op::Op, r#type::Typed};
                let ctx = scope.ctx_mut();
                let ptx = $crate::shared::closure_inference_hack::<$ty, String>(self, ctx, $ptx);
                let op = $crate::cuda::ptx::InlinePtxOp::new(
                    ctx,
                    Some(self.get_result(ctx).get_type(ctx)),
                    ptx,
                    self.get_operation().operands(ctx),
                );
                scope.register(&op);
                vec![op.result(ctx).unwrap()]
            }
        }
    };
    ($ty: ty, $ptx: expr) => {
        ptx_with_out!($ty, $ptx, |_, _| true);
    };
}
pub(super) use ptx_with_out;

op_includes!(Cuda, [SyncAsyncProxyOp] => "cuda/barrier");

cuda_op!(SyncOp, |op, ctx| {
    match op.scope(ctx).0 {
        SyncScope::Plane => "__syncwarp();\n",
        SyncScope::Cube | SyncScope::Device => "__syncthreads();\n",
        SyncScope::Unit => "",
    }
    .into()
});

cuda_op!(SyncAsyncProxyOp, |_, _| {
    "cuda::device::experimental::fence_proxy_async_shared_cta();".into()
});

pub(crate) const COMPLEX_HELPERS: &str = r#"
__device__ __host__ inline cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b) { return cuCaddf(a, b); }
__device__ __host__ inline cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }
__device__ __host__ inline cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
__device__ __host__ inline cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a, b); }
__device__ __host__ inline cuFloatComplex operator-(cuFloatComplex a) { return make_cuFloatComplex(-cuCrealf(a), -cuCimagf(a)); }
__device__ __host__ inline bool operator==(cuFloatComplex a, cuFloatComplex b) { return cuCrealf(a)==cuCrealf(b) && cuCimagf(a)==cuCimagf(b); }
__device__ __host__ inline bool operator!=(cuFloatComplex a, cuFloatComplex b) { return !(a==b); }
__device__ __host__ inline cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ __host__ inline cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }
__device__ __host__ inline cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
__device__ __host__ inline cuDoubleComplex operator/(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }
__device__ __host__ inline cuDoubleComplex operator-(cuDoubleComplex a) { return make_cuDoubleComplex(-cuCreal(a), -cuCimag(a)); }
__device__ __host__ inline bool operator==(cuDoubleComplex a, cuDoubleComplex b) { return cuCreal(a)==cuCreal(b) && cuCimag(a)==cuCimag(b); }
__device__ __host__ inline bool operator!=(cuDoubleComplex a, cuDoubleComplex b) { return !(a==b); }
__device__ __host__ inline float cubecl_abs(cuFloatComplex a) { return hypotf(cuCrealf(a), cuCimagf(a)); }
__device__ __host__ inline double cubecl_abs(cuDoubleComplex a) { return hypot(cuCreal(a), cuCimag(a)); }
__device__ __host__ inline cuFloatComplex cubecl_exp(cuFloatComplex a) { const float x=cuCrealf(a), y=cuCimagf(a), ex=expf(x); return make_cuFloatComplex(ex*cosf(y), ex*sinf(y)); }
__device__ __host__ inline cuDoubleComplex cubecl_exp(cuDoubleComplex a) { const double x=cuCreal(a), y=cuCimag(a), ex=exp(x); return make_cuDoubleComplex(ex*cos(y), ex*sin(y)); }
__device__ __host__ inline cuFloatComplex cubecl_log(cuFloatComplex a) { const float x=cuCrealf(a), y=cuCimagf(a); return make_cuFloatComplex(logf(hypotf(x,y)), atan2f(y,x)); }
__device__ __host__ inline cuDoubleComplex cubecl_log(cuDoubleComplex a) { const double x=cuCreal(a), y=cuCimag(a); return make_cuDoubleComplex(log(hypot(x,y)), atan2(y,x)); }
__device__ __host__ inline cuFloatComplex cubecl_sin(cuFloatComplex a) { const float x=cuCrealf(a), y=cuCimagf(a); return make_cuFloatComplex(sinf(x)*coshf(y), cosf(x)*sinhf(y)); }
__device__ __host__ inline cuDoubleComplex cubecl_sin(cuDoubleComplex a) { const double x=cuCreal(a), y=cuCimag(a); return make_cuDoubleComplex(sin(x)*cosh(y), cos(x)*sinh(y)); }
__device__ __host__ inline cuFloatComplex cubecl_cos(cuFloatComplex a) { const float x=cuCrealf(a), y=cuCimagf(a); return make_cuFloatComplex(cosf(x)*coshf(y), -sinf(x)*sinhf(y)); }
__device__ __host__ inline cuDoubleComplex cubecl_cos(cuDoubleComplex a) { const double x=cuCreal(a), y=cuCimag(a); return make_cuDoubleComplex(cos(x)*cosh(y), -sin(x)*sinh(y)); }
__device__ __host__ inline cuFloatComplex cubecl_sqrt(cuFloatComplex a) { const float x=cuCrealf(a), y=cuCimagf(a), r=hypotf(x,y); if(x>=0.0f){ const float re=sqrtf(0.5f*(r+x)); return make_cuFloatComplex(re,re==0.0f?0.0f:y/(2.0f*re)); } const float im=copysignf(sqrtf(0.5f*(r-x)),y); return make_cuFloatComplex(im==0.0f?0.0f:y/(2.0f*im),im); }
__device__ __host__ inline cuDoubleComplex cubecl_sqrt(cuDoubleComplex a) { const double x=cuCreal(a), y=cuCimag(a), r=hypot(x,y); if(x>=0.0){ const double re=sqrt(0.5*(r+x)); return make_cuDoubleComplex(re,re==0.0?0.0:y/(2.0*re)); } const double im=copysign(sqrt(0.5*(r-x)),y); return make_cuDoubleComplex(im==0.0?0.0:y/(2.0*im),im); }
__device__ __host__ inline cuFloatComplex cubecl_tanh(cuFloatComplex a) { const float x2=2.0f*cuCrealf(a), y2=2.0f*cuCimagf(a), d=coshf(x2)+cosf(y2); return make_cuFloatComplex(sinhf(x2)/d,sinf(y2)/d); }
__device__ __host__ inline cuDoubleComplex cubecl_tanh(cuDoubleComplex a) { const double x2=2.0*cuCreal(a), y2=2.0*cuCimag(a), d=cosh(x2)+cos(y2); return make_cuDoubleComplex(sinh(x2)/d,sin(y2)/d); }
__device__ __host__ inline cuFloatComplex cubecl_powf(cuFloatComplex a, cuFloatComplex b) { return cubecl_exp(b*cubecl_log(a)); }
__device__ __host__ inline cuDoubleComplex cubecl_powf(cuDoubleComplex a, cuDoubleComplex b) { return cubecl_exp(b*cubecl_log(a)); }
"#;
