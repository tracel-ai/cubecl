use crate::{
    cuda::{CudaDialect, arch::CudaArchitecture},
    shared::{
        Fragment, FragmentIdent, FragmentLayout, SupportedWmmaCombinations, WmmaCompiler,
        WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu};

const WMMA_NAMESPACE: &str = "nvcuda::wmma";
const WMMA_MINIMUM_VERSION: u32 = 70;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaWmmaCompiler {}

impl WmmaCompiler<CudaDialect> for CudaWmmaCompiler {
    type Architecture = CudaArchitecture;

    fn wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }

    fn deftypes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment_ident(
        ident: &FragmentIdent<CudaDialect>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(WMMA_NAMESPACE, ident, f)
    }

    fn compile_fragment_layout(
        layout: &FragmentLayout<CudaDialect>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(WMMA_NAMESPACE, layout, f)
    }

    fn compile_fragment(
        fragment: &Fragment<CudaDialect>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment(WMMA_NAMESPACE, fragment, f)
    }

    fn compile_instruction(
        instruction: &WmmaInstruction<CudaDialect>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_instruction(WMMA_NAMESPACE, instruction, f)
    }

    fn supported_wmma_combinations(arch: &Self::Architecture) -> SupportedWmmaCombinations {
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.version >= WMMA_MINIMUM_VERSION {
            let tdims = vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)];
            // Types fully supported.
            let types = vec![
                (
                    gpu::Elem::Float(gpu::FloatKind::F16), // a
                    gpu::Elem::Float(gpu::FloatKind::F16), // b
                    gpu::Elem::Float(gpu::FloatKind::F16), // c
                ),
                (
                    gpu::Elem::Float(gpu::FloatKind::F16),
                    gpu::Elem::Float(gpu::FloatKind::F16),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::Elem::Float(gpu::FloatKind::BF16),
                    gpu::Elem::Float(gpu::FloatKind::BF16),
                    gpu::Elem::Float(gpu::FloatKind::F32),
                ),
            ];
            let combinations: SupportedWmmaCombinations = types
                .into_iter()
                .map(|(a, b, c)| {
                    let dimensions = tdims.clone();
                    (a, b, c, dimensions)
                })
                .collect();
            result.extend(combinations);
            result.push((
                gpu::Elem::Float(gpu::FloatKind::TF32),
                gpu::Elem::Float(gpu::FloatKind::TF32),
                gpu::Elem::Float(gpu::FloatKind::F32),
                vec![(16, 16, 8)],
            ));
        }
        result
    }
}
