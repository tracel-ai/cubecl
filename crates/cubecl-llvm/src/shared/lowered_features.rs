//! What a device advertises, narrowed to what the LLVM backend lowers for it.
//!
//! A GPU runtime's properties come from its C++ backend, which gained each generation's hardware
//! features as they shipped. The LLVM backend runs ordinary kernels: arithmetic, memory, shared
//! memory, the plane operations, the two barriers and the matrix instructions it has register
//! shapes for, plus Hopper's TMA and warpgroup MMA on NVPTX. Everything else is taken away here rather than left to fail at compile time,
//! because a consumer picks its algorithm off these properties — cubek's matmul selectors ask for
//! `mma` before they ask anything else — and an advertisement that cannot be honoured is a launch
//! that fails rather than one that falls back. The CPU runtime builds its properties from what
//! this backend lowers, so it has nothing to narrow.

#[cfg(feature = "amdgpu")]
use cubecl_core::ir::amd::AmdWmma;
use cubecl_core::ir::{ComplexKind, DeviceProperties, ElemType, FloatKind, OpaqueType};
#[cfg(feature = "nvptx")]
use cubecl_core::ir::{IntKind, UIntKind};

const HALF: ElemType = ElemType::Float(FloatKind::F16);
const BF16: ElemType = ElemType::Float(FloatKind::BF16);

/// The GPU target a device's features are narrowed for, with what its lowering depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuTarget {
    #[cfg(feature = "nvptx")]
    Nvptx,
    /// `wmma` is the part's WMMA generation, `None` on a part without one.
    #[cfg(feature = "amdgpu")]
    AmdGpu { wmma: Option<AmdWmma> },
}

/// Narrows `props` to what the LLVM backend lowers for `target`: the matrix forms that target
/// has register shapes for, and nothing that neither target lowers.
pub fn restrict_features(props: &mut DeviceProperties, target: GpuTarget) {
    match target {
        #[cfg(feature = "nvptx")]
        GpuTarget::Nvptx => {
            keep_nvptx_matrix_forms(props);
            keep_nvptx_tma(props);
        }
        #[cfg(feature = "amdgpu")]
        GpuTarget::AmdGpu { wmma } => {
            keep_amdgpu_matrix_forms(props, wmma);
            remove_tma(props);
        }
    }
    restrict_common(props);
}

/// Keeps TMA where the device has it: the tiled and im2col tensor copies, the bulk groups, and
/// the cube barriers they complete on. Every barrier operation lowers to an `mbarrier`
/// instruction from Hopper, so a device without TMA keeps no barrier either. `im2colWide` is
/// Blackwell's and has no lowering.
#[cfg(feature = "nvptx")]
fn keep_nvptx_tma(props: &mut DeviceProperties) {
    use cubecl_core::ir::features::Tma;

    if !props.features.tma.contains(Tma::Base) {
        remove_tma(props);
        return;
    }
    props.features.tma.remove(Tma::Im2colWide);
}

/// No TMA, and no `mbarrier` behind it.
fn remove_tma(props: &mut DeviceProperties) {
    props.features.tma = Default::default();
    props.features.types.opaque.remove(&OpaqueType::TensorMap);
    props.features.types.opaque.remove(&OpaqueType::Barrier);
}

/// An accumulator the half-precision matrix instructions of NVPTX write.
#[cfg(feature = "nvptx")]
fn is_half_or_single(ty: ElemType) -> bool {
    matches!(
        ty,
        ElemType::Float(FloatKind::F16) | ElemType::Float(FloatKind::F32)
    )
}

/// Keeps the matrix forms NVPTX has register shapes for.
#[cfg(feature = "nvptx")]
fn keep_nvptx_matrix_forms(props: &mut DeviceProperties) {
    // Both matrix families support f16, bf16 and TF32 operands. TF32 uses FP32 storage,
    // rounded casts, and opaque i32 registers at the NVVM matrix call boundary; bf16 packs two
    // `bfloat` lanes to an i32 register and accumulates in f32. Keep only the TF32 geometries
    // NVVM implements.
    let byte = |ty: ElemType| {
        matches!(
            ty,
            ElemType::Int(IntKind::I8) | ElemType::UInt(UIntKind::U8)
        )
    };

    let tf32 = ElemType::Float(FloatKind::TF32);
    let f32 = ElemType::Float(FloatKind::F32);
    let matmul = &mut props.features.matmul;
    matmul.cmma.retain(|config| {
        let half =
            config.a_type == HALF && config.b_type == HALF && is_half_or_single(config.cd_type);
        let bf16 = config.a_type == BF16 && config.b_type == BF16 && config.cd_type == f32;
        let tf32 = config.a_type == tf32
            && config.b_type == tf32
            && config.cd_type == f32
            && (config.m, config.n, config.k) == (16, 16, 8);
        half || bf16 || tf32
    });
    matmul.mma.retain(|config| {
        let floats = (config.a_type == HALF || config.a_type == BF16)
            && config.b_type == config.a_type
            && config.cd_type == ElemType::Float(FloatKind::F32);
        // The four signed/unsigned pairings are four instructions over the same registers, so
        // the operands are taken independently.
        let integers = byte(config.a_type)
            && byte(config.b_type)
            && config.cd_type == ElemType::Int(IntKind::I32);
        let tf32 = config.a_type == tf32
            && config.b_type == tf32
            && config.cd_type == f32
            && config.m == 16
            && config.n == 8
            && matches!(config.k, 4 | 8);
        floats || integers || tf32
    });
    // The manual `mma.sync` family, `ldmatrix` and `stmatrix` are advertised as lowered;
    // `test_cmma_manual` checks them element by element.
}

/// Keeps the matrix forms AMDGPU lowers. The matrix lowering is RDNA's WMMA with `f16` or `bf16`
/// operands accumulating in their own type or `f32`: CDNA's MFMA and the integer and fp8 WMMA
/// forms have none. The plane operations work under
/// divergence, since they read the active lanes from `exec`, so they are left as advertised.
#[cfg(feature = "amdgpu")]
fn keep_amdgpu_matrix_forms(props: &mut DeviceProperties, wmma: Option<AmdWmma>) {
    let lowered = |a: ElemType, b: ElemType, cd: ElemType| {
        let operands = (a == HALF || a == BF16) && b == a;
        let accumulator = cd == a || cd == ElemType::Float(FloatKind::F32);
        wmma.is_some() && operands && accumulator
    };
    let matmul = &mut props.features.matmul;
    matmul
        .cmma
        .retain(|config| lowered(config.a_type, config.b_type, config.cd_type));
    matmul
        .mma
        .retain(|config| lowered(config.a_type, config.b_type, config.cd_type));
}

/// What neither GPU target lowers.
fn restrict_common(props: &mut DeviceProperties) {
    // The cube-level matrix API, and the scaled instructions with their `block_scale` operands.
    let matmul = &mut props.features.matmul;
    matmul.cube_mma = Default::default();
    matmul.scaled_mma = Default::default();
    matmul.cmma_tensor_addressing = false;
    if matmul.cmma.is_empty() && matmul.mma.is_empty() {
        props.hardware.num_tensor_cores = None;
        props.hardware.min_tensor_cores_dim = None;
    }

    // No clusters, and no `memcpy_async`, whose unit barriers have no lowering.
    props.features.cube_cluster = false;
    props.features.copy_async = false;

    // `bf16` atomics have no test on any target this backend lowers for, so they are not
    // offered until one checks the read-modify-write LLVM expands them into.
    props
        .features
        .types
        .atomic
        .retain(|ty, _| ty.elem_type() != BF16);

    // Complex arithmetic is lowered by the C++ backends, not by this one.
    props.features.types.complex.clear();
    for kind in [ComplexKind::C32, ComplexKind::C64] {
        props.features.types.elem.remove(&ElemType::Complex(kind));
    }

    // Vectorized float atomics: the shared atomic lowering handles the scalar widths, and a
    // vector `atomicrmw` is not one instruction on either target.
    props
        .features
        .types
        .atomic
        .retain(|ty, _| ty.vector_size() == 1);
}

#[cfg(all(test, feature = "amdgpu"))]
mod tests {
    use super::*;
    use crate::shared::offline_kernels::device_properties;
    use cubecl_core::ir::{
        IntKind, Type,
        amd::GfxArch,
        features::{AtomicUsage, MmaConfig, TypeUsage},
    };

    /// A part with no WMMA has no matrix lowering at all, so it must offer none.
    #[test]
    fn a_part_without_wmma_advertises_no_matrix_form() {
        let props = restricted_for("gfx90a");
        assert!(props.features.matmul.cmma.is_empty());
        assert!(props.features.matmul.mma.is_empty());
        assert_eq!(props.hardware.num_tensor_cores, None);
    }

    /// The lowering has register shapes for `f16` and `bf16` operands alone: the integer and fp8
    /// forms rocWMMA advertises would fail to compile.
    #[test]
    fn rdna_keeps_the_half_precision_forms_only() {
        let props = restricted_for("gfx1201");
        for kept in [&props.features.matmul.cmma, &props.features.matmul.mma] {
            assert_eq!(kept.len(), 3, "{kept:?}");
            assert!(
                kept.iter()
                    .all(|config| config.a_type == HALF || config.a_type == BF16),
                "{kept:?}"
            );
        }
    }

    /// `bf16` lowers as LLVM `bfloat`, arithmetic and conversions alike; its atomics are not
    /// offered.
    #[test]
    fn bf16_is_offered_without_its_atomics() {
        let bf16 = ElemType::Float(FloatKind::BF16);
        let mut props = (*device_properties(32)).clone();
        props.register_type_usage(bf16, TypeUsage::all());
        props.register_atomic_type_usage(Type::atomic(bf16), AtomicUsage::all());
        restrict_features(&mut props, GpuTarget::AmdGpu { wmma: None });
        assert!(props.features.types.elem.contains_key(&bf16));
        assert!(
            props
                .features
                .types
                .atomic
                .keys()
                .all(|ty| ty.elem_type() != bf16)
        );
    }

    fn restricted_for(arch: &str) -> DeviceProperties {
        let mut props = advertising_every_matrix_form();
        let wmma = GfxArch::parse(arch).wmma();
        restrict_features(&mut props, GpuTarget::AmdGpu { wmma });
        props
    }

    fn config(a: ElemType, cd: ElemType) -> MmaConfig {
        MmaConfig {
            a_type: a,
            b_type: a,
            cd_type: cd,
            m: 16,
            n: 16,
            k: 16,
        }
    }

    fn advertising_every_matrix_form() -> DeviceProperties {
        let mut props = (*device_properties(32)).clone();
        let f32 = ElemType::Float(FloatKind::F32);
        let forms = [
            config(HALF, f32),
            config(HALF, HALF),
            config(ElemType::Float(FloatKind::BF16), f32),
            config(ElemType::Int(IntKind::I8), ElemType::Int(IntKind::I32)),
        ];
        props.features.matmul.cmma.extend(forms);
        props.features.matmul.mma.extend(forms);
        props
    }
}

#[cfg(all(test, feature = "nvptx"))]
mod nvptx_tests {
    use super::*;
    use crate::shared::offline_kernels::device_properties;
    use cubecl_core::ir::features::MmaConfig;

    #[test]
    fn tf32_keeps_only_supported_matrix_shapes() {
        let mut props = (*device_properties(32)).clone();
        let tf32 = ElemType::Float(FloatKind::TF32);
        let f32 = ElemType::Float(FloatKind::F32);
        let config = |m, n, k| MmaConfig {
            a_type: tf32,
            b_type: tf32,
            cd_type: f32,
            m,
            n,
            k,
        };
        let forms = [
            config(16, 16, 8),
            config(16, 8, 4),
            config(16, 8, 8),
            config(16, 8, 16),
        ];
        props.features.matmul.cmma.extend(forms);
        props.features.matmul.mma.extend(forms);
        restrict_features(&mut props, GpuTarget::Nvptx);
        assert_eq!(props.features.matmul.cmma.len(), 1);
        assert!(props.features.matmul.cmma.contains(&config(16, 16, 8)));
        assert_eq!(props.features.matmul.mma.len(), 2);
        assert!(props.features.matmul.mma.contains(&config(16, 8, 4)));
        assert!(props.features.matmul.mma.contains(&config(16, 8, 8)));
    }

    /// `bf16` operands accumulate in `f32` only: no WMMA or `mma.sync` form writes `bf16`.
    #[test]
    fn bf16_keeps_its_f32_accumulator_forms() {
        let mut props = (*device_properties(32)).clone();
        let f32 = ElemType::Float(FloatKind::F32);
        let config = |cd_type, m, n, k| MmaConfig {
            a_type: BF16,
            b_type: BF16,
            cd_type,
            m,
            n,
            k,
        };
        props.features.matmul.cmma.extend([
            config(f32, 16, 16, 16),
            config(f32, 32, 8, 16),
            config(BF16, 16, 16, 16),
        ]);
        props
            .features
            .matmul
            .mma
            .extend([config(f32, 16, 8, 16), config(BF16, 16, 8, 16)]);
        restrict_features(&mut props, GpuTarget::Nvptx);
        assert_eq!(props.features.matmul.cmma.len(), 2);
        assert!(props.features.matmul.cmma.iter().all(|c| c.cd_type == f32));
        assert_eq!(props.features.matmul.mma.len(), 1);
        assert!(props.features.matmul.mma.contains(&config(f32, 16, 8, 16)));
    }
}

#[cfg(all(test, feature = "nvptx"))]
mod nvptx_tma_tests {
    use super::*;
    use crate::shared::offline_kernels::device_properties;
    use cubecl_core::ir::features::Tma;

    fn restricted(tma: &[Tma]) -> DeviceProperties {
        let mut props = (*device_properties(32)).clone();
        props.register_opaque_type(OpaqueType::Barrier);
        if !tma.is_empty() {
            props.register_opaque_type(OpaqueType::TensorMap);
        }
        for &feature in tma {
            props.features.tma.insert(feature);
        }
        restrict_features(&mut props, GpuTarget::Nvptx);
        props
    }

    /// Hopper keeps TMA and the barriers its loads complete on, but not Blackwell's
    /// `im2colWide`, which has no lowering.
    #[test]
    fn a_device_with_tma_keeps_it() {
        let props = restricted(&[Tma::Base, Tma::Im2colWide]);
        assert!(props.features.tma.contains(Tma::Base));
        assert!(!props.features.tma.contains(Tma::Im2colWide));
        assert!(props.features.types.opaque.contains(&OpaqueType::TensorMap));
        assert!(props.features.types.opaque.contains(&OpaqueType::Barrier));
        assert!(!props.features.copy_async);
    }

    /// Every barrier operation is an `mbarrier` instruction from Hopper, so an older device
    /// keeps no barrier.
    #[test]
    fn a_device_without_tma_keeps_no_barrier() {
        let props = restricted(&[]);
        assert!(props.features.tma.is_empty());
        assert!(!props.features.types.opaque.contains(&OpaqueType::Barrier));
    }
}
