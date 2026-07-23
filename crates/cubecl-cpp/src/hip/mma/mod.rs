pub mod manual;
pub mod rocwmma_compiler;
pub mod wmma_intrinsics_compiler;

use cubecl_core::{
    cmma::MatrixType,
    ir::{ContextExt, dialect::matrix::*},
};
use pliron::{
    builtin::op_interfaces::OneResultInterface, context::Context, derive::type_interface_impl,
    r#type::Typed,
};

use rocwmma_compiler::*;
pub use wmma_intrinsics_compiler::*;

use crate::{
    hip::{arch::AMDArchitecture, hip_op, ty::hip_ty},
    shared::{DeclareMatrixOp, SupportedMmaCombinations, ty::TypeToCPP, wmma_api_base},
    target::Hip,
};

const WMMA_NAMESPACE: &str = "rocwmma";

#[derive(Clone, Copy, Debug)]
pub enum HipCmmaCompiler {
    RocWmma,
    Intrinsics,
}

impl HipCmmaCompiler {
    pub fn supported_cmma_combinations(&self, arch: &AMDArchitecture) -> SupportedMmaCombinations {
        match self {
            HipCmmaCompiler::RocWmma => supported_wmma_combinations_rocwmma(arch),
            HipCmmaCompiler::Intrinsics => supported_wmma_combinations_intrinsic(arch),
        }
    }

    pub fn imports(&self) -> String {
        match self {
            HipCmmaCompiler::RocWmma => compile_rocwmma_includes(),
            HipCmmaCompiler::Intrinsics => String::new(),
        }
    }

    // These used to be based on flags but extra type defs don't actually hurt anything so there's
    // no point adding complexity
    pub fn type_definitions(&self) -> String {
        match self {
            HipCmmaCompiler::RocWmma => String::new(),
            HipCmmaCompiler::Intrinsics => r#"
typedef __bf16 bhalf8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bhalf16_t __attribute__((ext_vector_type(16)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));
        "#
            .into(),
        }
    }
}

impl HipCmmaExt for Context {}
pub trait HipCmmaExt: ContextExt {
    fn hip_cmma(&self) -> HipCmmaCompiler {
        *self.aux_ty::<HipCmmaCompiler>()
    }
    fn set_hip_cmma(&mut self, value: HipCmmaCompiler) {
        self.set_aux_ty(value);
    }
}

hip_ty!(MatrixType, |ty, ctx| match ctx.hip_cmma() {
    HipCmmaCompiler::RocWmma => wmma_api_base::compile_matrix(ctx, ty, WMMA_NAMESPACE),
    HipCmmaCompiler::Intrinsics => compile_fragment_intrinsic(ctx, ty),
});

hip_op!(DeclareMatrixOp, |op, ctx| {
    wmma_api_base::compile_matrix_declaration(
        ctx,
        op.get_result(ctx),
        op.value_ty(ctx).get_type(ctx),
    )
});

hip_op!(FillOp, |op, ctx| match ctx.hip_cmma() {
    HipCmmaCompiler::RocWmma => wmma_api_base::fill(ctx, op, WMMA_NAMESPACE),
    HipCmmaCompiler::Intrinsics => compile_fill_intrinsic(ctx, op),
});

hip_op!(LoadOp, |op, ctx| match ctx.hip_cmma() {
    HipCmmaCompiler::RocWmma => wmma_api_base::load(ctx, op, WMMA_NAMESPACE),
    HipCmmaCompiler::Intrinsics => compile_load_intrinsic(ctx, op),
});

hip_op!(StoreOp, |op, ctx| match ctx.hip_cmma() {
    HipCmmaCompiler::RocWmma => wmma_api_base::store(ctx, op, WMMA_NAMESPACE),
    HipCmmaCompiler::Intrinsics => compile_store_intrinsic(ctx, op),
});

hip_op!(MultiplyAccumulateOp, |op, ctx| match ctx.hip_cmma() {
    HipCmmaCompiler::RocWmma => wmma_api_base::execute(ctx, op, WMMA_NAMESPACE),
    HipCmmaCompiler::Intrinsics => compile_execute_intrinsic(ctx, op),
});

hip_op!(CastOp, |op, ctx| match ctx.hip_cmma() {
    HipCmmaCompiler::RocWmma => wmma_api_base::cast(ctx, op),
    HipCmmaCompiler::Intrinsics => compile_cast_intrinsic(ctx, op),
});
