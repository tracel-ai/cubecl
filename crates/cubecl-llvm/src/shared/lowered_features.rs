//! What a device advertises, narrowed to what the LLVM backend lowers for it.
//!
//! A GPU runtime's properties come from its C++ backend, which gained each generation's hardware
//! features as they shipped. The LLVM backend runs ordinary kernels: arithmetic, memory, shared
//! memory, the plane operations, the two barriers and the matrix instructions it has register
//! shapes for. Everything else is taken away here rather than left to fail at compile time,
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

/// An accumulator the half-precision matrix instructions of both targets write.
fn is_half_or_single(ty: ElemType) -> bool {
    matches!(
        ty,
        ElemType::Float(FloatKind::F16) | ElemType::Float(FloatKind::F32)
    )
}

/// Keeps the matrix forms NVPTX has register shapes for, and takes away what neither target
/// lowers.
#[cfg(feature = "nvptx")]
pub fn restrict_nvptx_features(props: &mut DeviceProperties) {
    // Both matrix families are lowered: the cooperative one through `wmma`, the manual one
    // through `mma.sync`. Each is narrowed to the element types its lowering has register
    // shapes for -- `f16` operands throughout, plus the narrow integers on the manual side,
    // which pass their registers as opaque words. `bf16` and `tf32` are in neither for the
    // same reason `bf16` is dropped in `restrict_common`: the dialect this backend lowers
    // through has no type for them, so there is nothing to put in a register.
    let byte = |ty: ElemType| {
        matches!(
            ty,
            ElemType::Int(IntKind::I8) | ElemType::UInt(UIntKind::U8)
        )
    };

    let matmul = &mut props.features.matmul;
    matmul.cmma.retain(|config| {
        config.a_type == HALF && config.b_type == HALF && is_half_or_single(config.cd_type)
    });
    matmul.mma.retain(|config| {
        let floats = config.a_type == HALF
            && config.b_type == HALF
            && config.cd_type == ElemType::Float(FloatKind::F32);
        // The four signed/unsigned pairings are four instructions over the same registers, so
        // the operands are taken independently.
        let integers = byte(config.a_type)
            && byte(config.b_type)
            && config.cd_type == ElemType::Int(IntKind::I32);
        floats || integers
    });
    // The manual `mma.sync` family, `ldmatrix` and `stmatrix` are advertised as lowered;
    // `test_cmma_manual` checks them element by element.

    restrict_common(props);
}

/// Keeps the matrix forms AMDGPU lowers, then removes what neither target lowers. The matrix
/// lowering is RDNA's WMMA with `f16` operands: CDNA's MFMA, the integer and fp8
/// WMMA forms and `bf16` (see `restrict_common`) have none. The plane operations work under
/// divergence, since they read the active lanes from `exec`, so they are left as advertised.
#[cfg(feature = "amdgpu")]
pub fn restrict_amdgpu_features(props: &mut DeviceProperties, wmma: Option<AmdWmma>) {
    let lowered = |a: ElemType, b: ElemType, cd: ElemType| {
        wmma.is_some() && a == HALF && b == HALF && is_half_or_single(cd)
    };
    let matmul = &mut props.features.matmul;
    matmul
        .cmma
        .retain(|config| lowered(config.a_type, config.b_type, config.cd_type));
    matmul
        .mma
        .retain(|config| lowered(config.a_type, config.b_type, config.cd_type));

    restrict_common(props);
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

    // No TMA, no clusters, no async copy, and no `mbarrier` behind them.
    props.features.tma = Default::default();
    props.features.cube_cluster = false;
    props.features.copy_async = false;
    props.features.types.opaque.remove(&OpaqueType::TensorMap);
    props.features.types.opaque.remove(&OpaqueType::Barrier);

    // `bf16` has no type in the LLVM dialect this backend lowers through -- pliron has
    // `builtin.fp16`, `fp32` and `fp64` and nothing between -- so a `bf16` kernel compiles to
    // something that quietly computes zeros. Until it is either given a type or carried as an
    // `i16` the way the minifloats are, it must not be offered.
    let bf16 = ElemType::Float(FloatKind::BF16);
    props.features.types.elem.remove(&bf16);
    props
        .features
        .types
        .atomic
        .retain(|ty, _| ty.elem_type() != bf16);

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
    use cubecl_core::ir::{IntKind, amd::GfxArch, features::MmaConfig};

    /// A part with no WMMA has no matrix lowering at all, so it must offer none.
    #[test]
    fn a_part_without_wmma_advertises_no_matrix_form() {
        let props = restricted_for("gfx90a");
        assert!(props.features.matmul.cmma.is_empty());
        assert!(props.features.matmul.mma.is_empty());
        assert_eq!(props.hardware.num_tensor_cores, None);
    }

    /// The lowering has register shapes for `f16` operands alone: the integer, fp8 and `bf16`
    /// forms rocWMMA advertises would fail to compile.
    #[test]
    fn rdna_keeps_the_half_precision_forms_only() {
        let props = restricted_for("gfx1201");
        for kept in [&props.features.matmul.cmma, &props.features.matmul.mma] {
            assert_eq!(kept.len(), 2, "{kept:?}");
            assert!(kept.iter().all(|config| config.a_type == HALF), "{kept:?}");
        }
    }

    fn restricted_for(arch: &str) -> DeviceProperties {
        let mut props = advertising_every_matrix_form();
        restrict_amdgpu_features(&mut props, GfxArch::parse(arch).wmma());
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
