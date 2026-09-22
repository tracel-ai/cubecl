//! What a device advertises, narrowed to what the LLVM backend lowers for it.

use cubecl_core::ir::{
    ComplexKind, DeviceProperties, ElemType, FloatKind, IntKind, OpaqueType, UIntKind,
    features::Plane,
};

/// A GPU target, with what its lowering depends on about the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmGpuTarget {
    #[cfg(feature = "nvptx")]
    Nvptx,
}

impl LlvmGpuTarget {
    /// Takes away every feature `props` advertises that this target does not lower.
    ///
    /// A runtime's properties come from its C++ backend, which gained each generation's
    /// hardware features as they shipped. A consumer picks its algorithm off these properties
    /// — cubek's matmul selectors ask for `mma` before anything else — so an advertisement the
    /// LLVM backend cannot honour is a launch that fails rather than one that falls back.
    pub fn restrict(self, props: &mut DeviceProperties) {
        match self {
            #[cfg(feature = "nvptx")]
            LlvmGpuTarget::Nvptx => restrict_nvptx(props),
        }
    }
}

/// Narrows what the device advertises to what the LLVM backend actually lowers.
///
/// A CUDA device's properties are the C++ backend's, which has had every generation of NVIDIA's
/// hardware features added to it as they shipped. The LLVM backend is at the point of running
/// ordinary kernels: arithmetic, memory, shared memory, the plane operations and the two
/// barriers. Everything it does not lower is taken away here rather than left to fail at
/// compile time, because a consumer picks its algorithm off these properties — cubek's matmul
/// selectors ask for `mma` before they ask anything else — and an advertisement that cannot be
/// honoured is a launch that fails rather than one that falls back.
///
/// Each of these comes back as its lowering lands; see the matrix and TMA work in the `nvptx`
/// module.
#[cfg(feature = "nvptx")]
fn restrict_nvptx(props: &mut DeviceProperties) {
    // Both matrix families are lowered: the cooperative one through `wmma`, the manual one
    // through `mma.sync`. Each is narrowed to the element types its lowering has register
    // shapes for -- `f16` operands throughout, plus the narrow integers on the manual side,
    // which pass their registers as opaque words. `bf16` and `tf32` are in neither for the
    // same reason `bf16` is dropped below: the dialect this backend lowers through has no type
    // for them, so there is nothing to put in a register.
    let half = ElemType::Float(FloatKind::F16);
    let byte = |ty: ElemType| {
        matches!(
            ty,
            ElemType::Int(IntKind::I8) | ElemType::UInt(UIntKind::U8)
        )
    };

    let matmul = &mut props.features.matmul;
    matmul.cmma.retain(|config| {
        config.a_type == half
            && config.b_type == half
            && matches!(
                config.cd_type,
                ElemType::Float(FloatKind::F16) | ElemType::Float(FloatKind::F32)
            )
    });
    matmul.mma.retain(|config| {
        let floats = config.a_type == half
            && config.b_type == half
            && config.cd_type == ElemType::Float(FloatKind::F32);
        // The four signed/unsigned pairings are four instructions over the same registers, so
        // the operands are taken independently.
        let integers = byte(config.a_type)
            && byte(config.b_type)
            && config.cd_type == ElemType::Int(IntKind::I32);
        floats || integers
    });
    // The manual `mma.sync` family, `ldmatrix` and `stmatrix` are advertised: the lowering is
    // correct, which `test_cmma_manual` checks element by element and cubek's
    // `multi_level::basic::plane_accelerated::*_mma` matmuls now agree with.
    //
    // Those matmuls did come out wrong here for a while, and it is worth saying why they were
    // not this backend's fault: they published an accumulator tile to shared memory and read it
    // back across the plane with no barrier, which works on a backend whose optimizer takes the
    // code at its word and does not survive one that does not. The barrier belongs in the
    // kernel and is now there.

    // Still on the manual side and still unimplemented: the cube-level API, and the scaled
    // instructions with their `block_scale` operands.
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

    // The shuffles go through `shfl.sync` with a full member mask, which requires the plane to
    // be converged. The C++ backend advertises this because its own plane lowering handles a
    // partial mask; until this one does, a diverged plane operation would be undefined rather
    // than merely slow.
    props.features.plane.remove(Plane::NonUniformControlFlow);

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
    // vector `atomicrmw` is not one instruction on this target.
    props
        .features
        .types
        .atomic
        .retain(|ty, _| ty.vector_size() == 1);
}
