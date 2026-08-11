use cubecl_core::{
    cmma::MatrixType,
    ir::{
        ContextExt,
        dialect::matrix::{CastOp, FillOp, LoadOp, MultiplyAccumulateOp, StoreOp},
        prelude::*,
    },
};

use crate::{
    cuda::{arch::CudaArchitecture, cuda_op, ty::cuda_ty},
    shared::{DeclareMatrixOp, SupportedMmaCombinations, wmma_api_base},
};

pub mod cuda_compiler;
pub mod manual;
pub mod ptx_wmma_compiler;

use cuda_compiler::*;
use ptx_wmma_compiler::*;

const WMMA_NAMESPACE: &str = "nvcuda::wmma";
const WMMA_MINIMUM_VERSION: u32 = 70;

// Maybe this should be a separate dialect but lets keep it simple for now

#[derive(Clone, Copy, Debug)]
pub enum CudaCmmaCompiler {
    Cpp,
    Ptx,
}

impl CudaCmmaCompiler {
    pub fn supported_cmma_combinations(&self, arch: &CudaArchitecture) -> SupportedMmaCombinations {
        match self {
            CudaCmmaCompiler::Cpp => supported_cmma_combinations_wmma(arch),
            CudaCmmaCompiler::Ptx => supported_cmma_combinations_ptx(arch),
        }
    }

    pub fn imports(&self) -> String {
        "#include <mma.h>\n".into()
    }
}

impl CudaCmmaExt for Context {}
pub trait CudaCmmaExt: ContextExt {
    fn cuda_cmma(&self) -> CudaCmmaCompiler {
        *self.aux_ty::<CudaCmmaCompiler>()
    }
    fn set_cuda_cmma(&mut self, value: CudaCmmaCompiler) {
        self.set_aux_ty(value);
    }
}

cuda_ty!(MatrixType, |ty, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::compile_matrix(ctx, ty, WMMA_NAMESPACE),
    CudaCmmaCompiler::Ptx => compile_matrix_ptx(ctx, ty),
});

cuda_op!(DeclareMatrixOp, |op, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::compile_matrix_declaration(
        ctx,
        op.get_result(ctx),
        op.value_ty(ctx).get_type(ctx),
    ),
    CudaCmmaCompiler::Ptx =>
        compile_matrix_declaration_ptx(ctx, op.get_result(ctx), op.value_ty(ctx).get_type(ctx)),
});

cuda_op!(FillOp, |op, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::fill(ctx, op, WMMA_NAMESPACE),
    CudaCmmaCompiler::Ptx => fill_ptx(ctx, op),
});

cuda_op!(LoadOp, |op, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::load(ctx, op, WMMA_NAMESPACE),
    CudaCmmaCompiler::Ptx => load_ptx(ctx, op),
});

cuda_op!(StoreOp, |op, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::store(ctx, op, WMMA_NAMESPACE),
    CudaCmmaCompiler::Ptx => store_ptx(ctx, op),
});

cuda_op!(MultiplyAccumulateOp, |op, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::execute(ctx, op, WMMA_NAMESPACE),
    CudaCmmaCompiler::Ptx => execute_ptx(ctx, op),
});

cuda_op!(CastOp, |op, ctx| match ctx.cuda_cmma() {
    CudaCmmaCompiler::Cpp => wmma_api_base::cast(ctx, op),
    CudaCmmaCompiler::Ptx => cast_ptx(ctx, op),
});
