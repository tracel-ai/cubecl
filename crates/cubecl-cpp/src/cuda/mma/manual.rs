use cubecl_core::{
    cmma::{MatrixIdent, MatrixType},
    ir::{
        ElemType, FloatKind, IntKind, UIntKind,
        dialect::matrix::{ColIndexOp, RowIndexOp},
        features::{MmaConfig, ScaledMmaConfig},
        interfaces::TypedExt,
        prelude::*,
    },
    prelude::*,
};
use itertools::Itertools;
use pliron::r#type::TypedHandle;

use cubecl_core::prelude::polyfills::mma::{col_index, row_index};

use crate::{
    cuda::arch::CudaArchitecture,
    shared::{
        Architecture, SupportedMmaCombinations, SupportedScaledMmaCombinations, lowering::LowerOp,
    },
    target::Cuda,
};

#[op_interface_impl]
impl LowerOp<Cuda> for RowIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
        let elems_per_reg = 32 / matrix.unpacked_elem_size_bits(scope.ctx());
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        let out = row_index::expand(scope, lane_id.into(), i.into(), elems_per_reg, matrix.ident);
        vec![out.value(scope)]
    }
}

#[op_interface_impl]
impl LowerOp<Cuda> for ColIndexOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let matrix = *self.matrix_ty(scope.ctx()).deref(scope.ctx());
        let elems_per_reg = 32 / matrix.unpacked_elem_size_bits(scope.ctx());
        let lane_id = self.lane_id(scope.ctx());
        let i = self.i(scope.ctx());
        let out = col_index::expand(scope, lane_id.into(), i.into(), elems_per_reg, matrix.ident);
        vec![out.value(scope)]
    }
}

pub fn supported_mma_combinations(arch: &CudaArchitecture) -> SupportedMmaCombinations {
    if !arch.tensor_cores {
        return vec![];
    }
    let mut result: SupportedMmaCombinations = vec![];
    // Higher than WMMA because we only support the newest shapes. Other shapes would make things
    // very complicated.
    // Also only use f32 accumulators for now
    if arch.get_version() >= 80 {
        result.extend([
            MmaConfig {
                a_type: ElemType::Float(FloatKind::F16),  // a
                b_type: ElemType::Float(FloatKind::F16),  // b
                cd_type: ElemType::Float(FloatKind::F32), // cd
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: ElemType::Float(FloatKind::BF16),
                b_type: ElemType::Float(FloatKind::BF16),
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 8,
                k: 16,
            },
            MmaConfig {
                a_type: ElemType::Float(FloatKind::TF32),
                b_type: ElemType::Float(FloatKind::TF32),
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 8,
                k: 8,
            },
            MmaConfig {
                a_type: ElemType::Int(IntKind::I8),
                b_type: ElemType::Int(IntKind::I8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::UInt(UIntKind::U8),
                b_type: ElemType::UInt(UIntKind::U8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::Int(IntKind::I8),
                b_type: ElemType::UInt(UIntKind::U8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            MmaConfig {
                a_type: ElemType::UInt(UIntKind::U8),
                b_type: ElemType::Int(IntKind::I8),
                cd_type: ElemType::Int(IntKind::I32),
                m: 16,
                n: 8,
                k: 32,
            },
            // TODO: u4/i4/b1, there's no types for them yet
        ]);
    }
    let fp8_types = [FloatKind::E4M3, FloatKind::E5M2];
    let fp6_fp4_types = [FloatKind::E3M2, FloatKind::E2M3, FloatKind::E2M1];
    if arch.get_version() >= 89 {
        let fp8_pairs = fp8_types.iter().cartesian_product(fp8_types.iter());
        result.extend(
            fp8_pairs.map(|(a_type, b_type)| {
                minifloat_mma_m16n8k32_accumulating_in_f32(*a_type, *b_type)
            }),
        );
    }
    // sm_120f: ptxas refuses an FP6 or FP4 operand of `mma.sync` outside the sm_120 family,
    // sm_100a included.
    if arch.get_version() >= 120 && arch.get_version() < 130 {
        let f8f6f4_types = fp8_types.iter().chain(fp6_fp4_types.iter());
        let pairs_with_an_fp6_or_fp4_operand = f8f6f4_types
            .clone()
            .cartesian_product(f8f6f4_types)
            .filter(|(a_type, b_type)| {
                fp6_fp4_types.contains(a_type) || fp6_fp4_types.contains(b_type)
            });
        result.extend(
            pairs_with_an_fp6_or_fp4_operand.map(|(a_type, b_type)| {
                minifloat_mma_m16n8k32_accumulating_in_f32(*a_type, *b_type)
            }),
        );
    }
    // Turing, not Volta: ptxas refuses `.m16n8k8` below sm_75, and sm_70 has
    // `m8n8k4` alone.
    //
    // Warning: this likely does not follow the same layout pattern as those after 80
    if arch.get_version() >= 75 && arch.get_version() < 80 {
        result.push(MmaConfig {
            a_type: ElemType::Float(FloatKind::F16),
            b_type: ElemType::Float(FloatKind::F16),
            cd_type: ElemType::Float(FloatKind::F32),
            m: 16,
            n: 8,
            k: 8,
        });
    }
    result
}

fn minifloat_mma_m16n8k32_accumulating_in_f32(a_type: FloatKind, b_type: FloatKind) -> MmaConfig {
    MmaConfig {
        a_type: ElemType::Float(a_type),
        b_type: ElemType::Float(b_type),
        cd_type: ElemType::Float(FloatKind::F32),
        m: 16,
        n: 8,
        k: 32,
    }
}

pub fn supported_scaled_mma_combinations(
    arch: &CudaArchitecture,
) -> SupportedScaledMmaCombinations {
    if !arch.tensor_cores {
        return vec![];
    }
    let mut result: SupportedScaledMmaCombinations = vec![];
    // sm_120f
    if arch.get_version() >= 120 && arch.get_version() < 130 {
        let f8f6f4_types = [
            FloatKind::E4M3,
            FloatKind::E5M2,
            FloatKind::E3M2,
            FloatKind::E2M3,
            FloatKind::E2M1,
        ];
        let combinations = f8f6f4_types
            .iter()
            .flat_map(|t1| f8f6f4_types.iter().map(move |t2| (t1, t2)));

        result.extend(combinations.map(|(t1, t2)| ScaledMmaConfig {
            a_type: ElemType::Float(*t1),
            b_type: ElemType::Float(*t2),
            cd_type: ElemType::Float(FloatKind::F32),
            scales_type: ElemType::Float(FloatKind::UE8M0),
            m: 16,
            n: 8,
            k: 32,
            scales_factor: 1,
        }));

        result.extend([
            ScaledMmaConfig {
                a_type: ElemType::Float(FloatKind::E2M1x2),
                b_type: ElemType::Float(FloatKind::E2M1x2),
                cd_type: ElemType::Float(FloatKind::F32),
                scales_type: ElemType::Float(FloatKind::UE8M0),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 2,
            },
            // Sign of scales is ignored
            ScaledMmaConfig {
                a_type: ElemType::Float(FloatKind::E2M1x2),
                b_type: ElemType::Float(FloatKind::E2M1x2),
                cd_type: ElemType::Float(FloatKind::F32),
                scales_type: ElemType::Float(FloatKind::E4M3),
                m: 16,
                n: 8,
                k: 64,
                scales_factor: 4,
            },
        ]);
    }
    result
}

pub fn contiguous_elements_cuda(
    ctx: &Context,
    ident: MatrixIdent,
    matrix: TypedHandle<MatrixType>,
) -> usize {
    let elem = matrix.deref(ctx).elem_ty;
    match ident {
        MatrixIdent::A | MatrixIdent::B => 32 / elem.size_bits(ctx),
        MatrixIdent::Accumulator => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::supported_mma_combinations;
    use crate::cuda::arch::CudaArchitecture;
    use cubecl_core::ir::{ElemType, FloatKind};
    use itertools::Itertools;

    /// A die with no tensor cores offers no MMA at all, so nothing downstream can pick a tile
    /// and measure the FP16 pipeline in units that claim tensor hardware.
    #[test]
    fn a_turing_die_without_tensor_cores_offers_no_mma() {
        let turing = |name: &str| CudaArchitecture {
            version: 75,
            tensor_cores: CudaArchitecture::has_tensor_cores(75, name),
        };

        assert!(supported_mma_combinations(&turing("NVIDIA GeForce GTX 1660 SUPER")).is_empty());
        assert!(!supported_mma_combinations(&turing("NVIDIA GeForce RTX 2060")).is_empty());
    }

    fn shapes(version: u32) -> Vec<(u32, u32, u32)> {
        supported_mma_combinations(&CudaArchitecture {
            version,
            tensor_cores: true,
        })
        .into_iter()
        .map(|config| (config.m, config.n, config.k))
        .collect()
    }

    /// `ptxas -arch=sm_70`: "Feature `.m16n8k8` requires `.target sm_75` or higher".
    /// Offering it there ends in `LLVM ERROR: Cannot select: intrinsic
    /// llvm.nvvm.mma.m16n8k8`, which aborts the process rather than failing a launch.
    #[test]
    fn volta_is_offered_no_mma_shape() {
        assert!(shapes(70).is_empty());
    }

    #[test]
    fn turing_keeps_its_one_shape() {
        assert_eq!(shapes(75), vec![(16, 8, 8)]);
    }

    /// The shapes from 80 on are their own set, reached by a separate branch.
    #[test]
    fn ampere_is_offered_more_than_turing() {
        assert!(shapes(80).len() > shapes(75).len());
    }

    const FP8_KINDS: [FloatKind; 2] = [FloatKind::E4M3, FloatKind::E5M2];
    const FP6_FP4_KINDS: [FloatKind; 3] = [FloatKind::E3M2, FloatKind::E2M3, FloatKind::E2M1];

    fn minifloat_operand_pairs(version: u32) -> Vec<(FloatKind, FloatKind)> {
        let is_minifloat = |ty: ElemType| match ty {
            ElemType::Float(kind) => FP8_KINDS.contains(&kind) || FP6_FP4_KINDS.contains(&kind),
            _ => false,
        };
        supported_mma_combinations(&CudaArchitecture {
            version,
            tensor_cores: true,
        })
        .into_iter()
        .filter(|config| is_minifloat(config.a_type) || is_minifloat(config.b_type))
        .map(|config| match (config.a_type, config.b_type) {
            (ElemType::Float(a_kind), ElemType::Float(b_kind)) => (a_kind, b_kind),
            pair => panic!("a minifloat MMA pairs two floats, got {pair:?}"),
        })
        .collect()
    }

    /// The kind-less FP8 `mma.sync` assembles from `sm_89` on, so every FP8 pairing is offered there.
    #[test]
    fn every_fp8_pair_is_offered_from_sm_89_on() {
        for version in [89, 90, 100, 103, 110, 120, 121] {
            let offered = minifloat_operand_pairs(version);
            for pair in FP8_KINDS.into_iter().cartesian_product(FP8_KINDS) {
                assert!(
                    offered.contains(&pair),
                    "sm_{version} does not offer {pair:?}"
                );
            }
        }
    }

    /// ptxas refuses an FP6 or FP4 operand of `mma.sync` outside the `sm_120` family.
    #[test]
    fn no_fp6_or_fp4_pair_is_offered_outside_the_sm_120_family() {
        for version in [89, 90, 100, 103, 110] {
            for (a_kind, b_kind) in minifloat_operand_pairs(version) {
                assert!(
                    FP8_KINDS.contains(&a_kind) && FP8_KINDS.contains(&b_kind),
                    "sm_{version} offers {a_kind:?} x {b_kind:?}"
                );
            }
        }
    }

    /// ptxas has no FP8 `mma.sync` below `sm_89`.
    #[test]
    fn no_fp8_pair_is_offered_below_sm_89() {
        for version in [75, 80, 86] {
            assert_eq!(minifloat_operand_pairs(version), vec![], "sm_{version}");
        }
    }
}
