use crate::shared::{Dialect, WmmaCompiler};

use super::{arch::CudaArchitecture, mma::CudaWmmaCompiler};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaDialect {}

impl WmmaCompiler<Self> for CudaDialect {
    type Architecture = CudaArchitecture;

    fn wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        CudaWmmaCompiler::wmma_includes(f)
    }

    fn deftypes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        CudaWmmaCompiler::deftypes(f)
    }

    fn local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        CudaWmmaCompiler::local_variables(f)
    }

    fn compile_fragment_ident(
        ident: &crate::shared::FragmentIdent<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_fragment_ident(ident, f)
    }

    fn compile_fragment_layout(
        layout: &crate::shared::FragmentLayout<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_fragment_layout(layout, f)
    }

    fn compile_fragment(
        fragment: &crate::shared::Fragment<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_fragment(fragment, f)
    }

    fn compile_instruction(
        instruction: &crate::shared::WmmaInstruction<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_instruction(instruction, f)
    }

    fn supported_wmma_combinations(
        arch: &Self::Architecture,
    ) -> crate::shared::SupportedWmmaCombinations {
        CudaWmmaCompiler::supported_wmma_combinations(arch)
    }
}

impl Dialect for CudaDialect {
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
    fn warp_shuffle(var: &str, source: &str) -> String {
        format!("__shfl_sync(-1, {var}, {source})")
    }
    fn warp_shuffle_xor(var: &str, offset: &str) -> String {
        format!("__shfl_xor_sync(-1, {var}, {offset})")
    }
    fn warp_shuffle_up(var: &str, offset: &str) -> String {
        format!("__shfl_up_sync(-1, {var}, {offset})")
    }
    fn warp_shuffle_down(var: &str, offset: &str) -> String {
        format!("__shfl_down_sync(-1, {var}, {offset})")
    }
    fn warp_all(var: &str) -> String {
        format!("__all_sync(-1, {var})")
    }
    fn warp_any(var: &str) -> String {
        format!("__any_sync(-1, {var})")
    }
    fn warp_ballot(out: &str) -> String {
        format!("__ballot_sync(-1, {out})")
    }
}
