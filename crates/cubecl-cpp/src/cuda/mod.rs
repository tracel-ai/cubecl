use crate::shared::{Dialect, IndexedVariable, Variable};
use std::marker::PhantomData;

use crate::shared::{
    wmma_api_base, Dialect, Fragment, FragmentIdent, FragmentLayout, Variable, WmmaCompiler, WmmaInstruction,
};

const WMMA_NAMESPACE: &str = "nvcuda::wmma";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaDialect<M: WmmaCompiler<Self>> {
    _wmma_compiler: PhantomData<M>,
}

impl<M: WmmaCompiler<Self>> Dialect for CudaDialect<M> {
    type WmmaCompiler = M;

    fn include_f16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_fp16.h>\n")
    }
    fn include_bf16(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_bf16.h>\n")
    }
    fn include_runtime(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <cuda_runtime.h>\n")
    }

    fn bfloat16_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("__nv_bfloat16")
    }
    fn bfloat162_type_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("__nv_bfloat162")
    }

    fn warp_shuffle(input: &IndexedVariable<Self>, id: &Variable<Self>) -> String {
        format!("__shfl_sync(-1, {input}, {id})")
    }
    fn warp_shuffle_xor(out: &IndexedVariable<Self>) -> String {
        format!("__shfl_xor_sync(-1, {out}, offset)")
    }
    fn warp_shuffle_down(out: &IndexedVariable<Self>) -> String {
        format!("__shfl_down_sync(-1, {out}, offset)")
    }
    fn warp_all(out: &IndexedVariable<Self>) -> String {
        format!("__all_sync(-1, {out})")
    }
    fn warp_any(out: &IndexedVariable<Self>) -> String {
        format!("__any_sync(-1, {out})")
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaWmmaCompiler {}

impl WmmaCompiler<CudaDialect<Self>> for CudaWmmaCompiler {
    fn includes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn deftypes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment_ident(
        ident: &FragmentIdent<CudaDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(WMMA_NAMESPACE, ident, f)
    }

    fn compile_fragment_layout(
        layout: &FragmentLayout<CudaDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(WMMA_NAMESPACE, layout, f)
    }

    fn compile_fragment(
        fragment: &Fragment<CudaDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment(WMMA_NAMESPACE, fragment, f)
    }

    fn compile_instruction(
        instruction: &WmmaInstruction<CudaDialect<Self>>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_instruction(WMMA_NAMESPACE, instruction, f)
    }
}
