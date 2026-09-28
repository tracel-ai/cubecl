//! What an NVIDIA compute capability is, for every backend that generates code for one.
//!
//! The CUDA runtime drives two of them, and the `-arch` a kernel is built for has to be the
//! same answer whichever one compiles it. The counterpart of [`amd`](crate::amd), kept
//! separate because the two name their devices nothing alike: AMD has a gfx string the driver
//! reports verbatim, NVIDIA a pair of integers.

use alloc::format;
use alloc::string::String;

/// A compute capability, as `cuDeviceGetAttribute` reports it.
///
/// Held as the packed two-digit form the CUDA tooling is written in — 8.6 is `86` — because
/// that is what `sm_86` and every threshold in the feature tables are spelled with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmArch {
    version: u32,
    /// Whether the die has tensor cores. Compute capability cannot say this on its own: the
    /// GTX-branded Turings report 7.5 like every other Turing and have none. Carried here so
    /// both backends read one answer rather than each deciding.
    tensor_cores: bool,
    /// Whether the architecture-specific (`a`) variant is the one to target. Blackwell and
    /// Hopper gate their tensor-core instructions behind it, and a `sm_90` kernel cannot
    /// reach them.
    arch_specific: bool,
}

impl SmArch {
    /// The architecture the runtime probed, with `tensor_cores` as it decided.
    pub fn new(version: u32, tensor_cores: bool) -> Self {
        Self {
            version,
            tensor_cores,
            // The same threshold the C++ backend's `--gpu-architecture` uses.
            arch_specific: version >= 90,
        }
    }

    /// The packed two-digit capability: 8.6 is `86`.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Whether the die has tensor cores at all.
    pub fn tensor_cores(&self) -> bool {
        self.tensor_cores
    }

    /// The `-mcpu` the NVPTX target machine takes, and the `.target` directive the emitted
    /// PTX carries — `sm_86`, or `sm_90a` where the architecture-specific variant exists.
    pub fn target_cpu(&self) -> String {
        if self.arch_specific {
            format!("sm_{}a", self.version)
        } else {
            format!("sm_{}", self.version)
        }
    }

    /// A warp is 32 lanes on every CUDA device there has ever been, which is why this is a
    /// constant here where the AMD side has to consult a table.
    pub fn plane_dim(&self) -> u32 {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// Hopper and Blackwell need the `a` suffix to reach their tensor-core instructions;
    /// everything before it must not carry one, since `sm_86a` is not a target that exists.
    #[test]
    fn only_hopper_and_later_ask_for_the_architecture_specific_variant() {
        assert_eq!(SmArch::new(86, true).target_cpu(), "sm_86".to_string());
        assert_eq!(SmArch::new(89, true).target_cpu(), "sm_89".to_string());
        assert_eq!(SmArch::new(90, true).target_cpu(), "sm_90a".to_string());
        assert_eq!(SmArch::new(120, true).target_cpu(), "sm_120a".to_string());
    }

    /// The tensor-core answer is the runtime's, not the version's: two dies reporting 7.5
    /// differ on it, and a backend that re-derived it from the version would disagree with
    /// the features the runtime advertised.
    #[test]
    fn the_tensor_core_answer_is_carried_not_rederived() {
        assert!(!SmArch::new(75, false).tensor_cores());
        assert!(SmArch::new(75, true).tensor_cores());
    }
}
