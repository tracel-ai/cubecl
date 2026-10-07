use super::WMMA_MINIMUM_VERSION;
use crate::{
    cuda::arch::CudaArchitecture,
    shared::{Architecture, SupportedMmaCombinations},
};
use cubecl_core::ir::{ElemType, FloatKind, IntKind, UIntKind, features::MmaConfig};
use itertools::Itertools;

pub(super) fn supported_cmma_combinations_wmma(
    arch: &CudaArchitecture,
) -> SupportedMmaCombinations {
    let mut result: SupportedMmaCombinations = vec![];
    if arch.get_version() >= WMMA_MINIMUM_VERSION && arch.tensor_cores {
        let tdims = [(16, 16, 16), (32, 8, 16), (8, 32, 16)];
        // Types fully supported.
        let types = vec![
            (
                ElemType::Float(FloatKind::F16), // m
                ElemType::Float(FloatKind::F16), // n
                ElemType::Float(FloatKind::F16), // k
            ),
            (
                ElemType::Float(FloatKind::F16),
                ElemType::Float(FloatKind::F16),
                ElemType::Float(FloatKind::F32),
            ),
            (
                ElemType::Int(IntKind::I8),
                ElemType::Int(IntKind::I8),
                ElemType::Int(IntKind::I32),
            ),
            (
                ElemType::UInt(UIntKind::U8),
                ElemType::UInt(UIntKind::U8),
                ElemType::Int(IntKind::I32),
            ),
        ];
        let combinations: SupportedMmaCombinations = types
            .into_iter()
            .cartesian_product(tdims.iter().copied())
            .map(|((a, b, c), (m, n, k))| MmaConfig {
                a_type: a,
                b_type: b,
                cd_type: c,
                m,
                n,
                k,
            })
            .collect();
        result.extend(combinations);
        // `bf16` and TF32 WMMA arrive with Ampere.
        if arch.get_version() >= 80 {
            result.extend(tdims.iter().map(|&(m, n, k)| MmaConfig {
                a_type: ElemType::Float(FloatKind::BF16),
                b_type: ElemType::Float(FloatKind::BF16),
                cd_type: ElemType::Float(FloatKind::F32),
                m,
                n,
                k,
            }));
            result.push(MmaConfig {
                a_type: ElemType::Float(FloatKind::TF32),
                b_type: ElemType::Float(FloatKind::TF32),
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 16,
                k: 8,
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::cuda::{arch::CudaArchitecture, mma::CudaCmmaCompiler};
    use cubecl_core::ir::{ElemType, FloatKind};

    fn offers_bf16(compiler: CudaCmmaCompiler, version: u32) -> bool {
        let arch = CudaArchitecture {
            version,
            tensor_cores: true,
        };
        compiler
            .supported_cmma_combinations(&arch)
            .iter()
            .any(|config| config.a_type == ElemType::Float(FloatKind::BF16))
    }

    /// `bf16` WMMA needs `sm_80`: offered on Volta or Turing, a kernel that trusts the feature
    /// fails to compile instead of falling back.
    #[test]
    fn bf16_wmma_starts_at_ampere() {
        for compiler in [CudaCmmaCompiler::Cpp, CudaCmmaCompiler::Ptx] {
            assert!(!offers_bf16(compiler, 70), "{compiler:?}");
            assert!(!offers_bf16(compiler, 75), "{compiler:?}");
            assert!(offers_bf16(compiler, 80), "{compiler:?}");
        }
    }
}
