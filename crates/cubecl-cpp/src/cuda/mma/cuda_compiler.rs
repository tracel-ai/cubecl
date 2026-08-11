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
    if arch.get_version() >= WMMA_MINIMUM_VERSION {
        let tdims = vec![(16, 16, 16), (32, 8, 16), (8, 32, 16)];
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
                ElemType::Float(FloatKind::BF16),
                ElemType::Float(FloatKind::BF16),
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
            .cartesian_product(tdims)
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
        if arch.get_version() >= 80 {
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
