use crate::{
    cuda::{arch::CudaArchitecture, CudaDialect},
    shared::{
        wmma_api_base, Fragment, FragmentIdent, FragmentLayout, SupportedWmmaCombinations,
        WmmaCompiler, WmmaInstruction,
    },
};
use cubecl_core::ir::{self as gpu};

const WMMA_NAMESPACE: &str = "nvcuda::wmma";
const WMMA_MINIMUM_VERSION: u32 = 70;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaWmmaCompiler {}

impl WmmaCompiler<CudaDialect<Self>> for CudaWmmaCompiler {
    type Architecture = CudaArchitecture;

    fn includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }

    fn deftypes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    fn supported_wmma_combinations(arch: &Self::Architecture) -> SupportedWmmaCombinations {
        let mut result: SupportedWmmaCombinations = vec![];
        if arch.version >= WMMA_MINIMUM_VERSION {
            //                                   m   n   k
            let tdims = vec![(16, 16, 16), (32, 16, 8), (8, 16, 32)];
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
                vec![(16, 8, 16)],
            ));
        }
        result
    }
}
