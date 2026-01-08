use super::{WMMA_MINIMUM_VERSION, WMMA_NAMESPACE};
use crate::{
    cuda::{
        CudaDialect,
        arch::CudaArchitecture,
        mma::{
            compile_manual_mma, compile_scaled_mma, supported_mma_combinations,
            supported_scaled_mma_combinations,
        },
    },
    shared::{
        Architecture, DialectWmmaCompiler, Flags, Fragment, FragmentIdent, FragmentLayout,
        ManualMma, SupportedMmaCombinations, SupportedScaledMmaCombinations, Variable,
        WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::ir::{self as gpu, features::MmaConfig};
use itertools::Itertools;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaWmmaCompiler {}

impl DialectWmmaCompiler<CudaDialect<Self>> for CudaWmmaCompiler {
    fn compile_wmma_includes(
        f: &mut std::fmt::Formatter<'_>,
        _flags: &Flags<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        f.write_str("#include <mma.h>\n")
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_declaration(f, var)
    }

    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &FragmentIdent<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_ident(f, WMMA_NAMESPACE, ident)
    }

    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &FragmentLayout<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_layout(f, WMMA_NAMESPACE, layout)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment(f, WMMA_NAMESPACE, fragment)
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_instruction(f, WMMA_NAMESPACE, instruction)
    }

    fn compile_manual_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<CudaDialect<Self>>,
    ) -> std::fmt::Result {
        compile_manual_mma(f, mma)
    }

    fn compile_scaled_mma(
        f: &mut std::fmt::Formatter<'_>,
        mma: ManualMma<CudaDialect<Self>>,
        scales_a: Variable<CudaDialect<Self>>,
        scales_b: Variable<CudaDialect<Self>>,
        scales_factor: u32,
    ) -> std::fmt::Result {
        compile_scaled_mma(f, mma, scales_a, scales_b, scales_factor)
    }

    fn supported_wmma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
        let mut result: SupportedMmaCombinations = vec![];
        if arch.get_version() >= WMMA_MINIMUM_VERSION {
            let tdims = vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)];
            // Types fully supported.
            let types = vec![
                (
                    gpu::ElemType::Float(gpu::FloatKind::F16), // m
                    gpu::ElemType::Float(gpu::FloatKind::F16), // n
                    gpu::ElemType::Float(gpu::FloatKind::F16), // k
                ),
                (
                    gpu::ElemType::Float(gpu::FloatKind::F16),
                    gpu::ElemType::Float(gpu::FloatKind::F16),
                    gpu::ElemType::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::ElemType::Float(gpu::FloatKind::BF16),
                    gpu::ElemType::Float(gpu::FloatKind::BF16),
                    gpu::ElemType::Float(gpu::FloatKind::F32),
                ),
                (
                    gpu::ElemType::Int(gpu::IntKind::I8),
                    gpu::ElemType::Int(gpu::IntKind::I8),
                    gpu::ElemType::Int(gpu::IntKind::I32),
                ),
                (
                    gpu::ElemType::UInt(gpu::UIntKind::U8),
                    gpu::ElemType::UInt(gpu::UIntKind::U8),
                    gpu::ElemType::Int(gpu::IntKind::I32),
                ),
            ];
            let combinations: SupportedMmaCombinations = types
                .into_iter()
                .cartesian_product(tdims)
                .map(|((a, b, c), (m, n, k))| MmaConfig {
                    a_type: a.into(),
                    b_type: b.into(),
                    cd_type: c.into(),
                    m,
                    n,
                    k,
                })
                .collect();
            result.extend(combinations);
            if arch.get_version() >= 80 {
                result.push(MmaConfig {
                    a_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                    b_type: gpu::ElemType::Float(gpu::FloatKind::TF32).into(),
                    cd_type: gpu::ElemType::Float(gpu::FloatKind::F32).into(),
                    m: 16,
                    n: 16,
                    k: 8,
                });
            }
        }
        result
    }

    fn supported_mma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
        supported_mma_combinations(arch)
    }

    fn supported_scaled_mma_combinations(
        arch: &CudaArchitecture,
    ) -> SupportedScaledMmaCombinations {
        supported_scaled_mma_combinations(arch)
    }
}
