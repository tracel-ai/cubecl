use std::fmt::Display;

use crate::shared::Architecture;

#[derive(Debug)]
pub struct CudaArchitecture {
    pub version: u32,
    /// Whether this die has tensor cores, which its compute capability does not say.
    ///
    /// TU116 and TU117 report 7.5 exactly like every other Turing and ship without them, so a
    /// version check alone advertises CMMA on a GTX 1660. The kernel then compiles, runs on
    /// the ordinary FP16 pipeline at about a fiftieth of the rate, and reports a number that
    /// looks comparable to one from a card that has the hardware.
    pub tensor_cores: bool,
}

impl CudaArchitecture {
    /// Decided by the marketing name, because CUDA exposes no attribute for it: there is no
    /// device property, and both dies share their compute capability with Turings that do have
    /// tensor cores.
    ///
    /// No GTX-branded Turing has them. Every Turing that does shipped as RTX, Titan RTX,
    /// Quadro RTX or Tesla, so the absence of tensor cores and the GTX brand coincide exactly
    /// on 7.5. The professional TU117 parts, T400 through T1000, also lack them and are not
    /// caught here; they keep reporting what they report today rather than being guessed at.
    pub fn has_tensor_cores(version: u32, name: &str) -> bool {
        // Volta brought them; nothing before it has any.
        if version < 70 {
            return false;
        }
        !(version == 75 && name.to_uppercase().contains("GTX"))
    }
}

#[cfg(test)]
mod tests {
    use super::CudaArchitecture;

    #[test]
    fn turing_without_tensor_cores_is_told_apart_from_turing_with_them() {
        // Same compute capability, opposite answers. This is the whole reason the name is read.
        assert!(!CudaArchitecture::has_tensor_cores(
            75,
            "NVIDIA GeForce GTX 1660 SUPER"
        ));
        assert!(CudaArchitecture::has_tensor_cores(
            75,
            "NVIDIA GeForce RTX 2060"
        ));
    }

    #[test]
    fn the_gtx_exception_applies_only_to_turing() {
        // A GTX at another compute capability is not a Turing and is not what this excludes.
        assert!(CudaArchitecture::has_tensor_cores(
            80,
            "NVIDIA A100-GTX-ish"
        ));
        // And Tesla T4 is TU104: 7.5, tensor cores, no GTX in the name.
        assert!(CudaArchitecture::has_tensor_cores(75, "Tesla T4"));
    }

    #[test]
    fn nothing_before_volta_has_them() {
        assert!(!CudaArchitecture::has_tensor_cores(
            61,
            "NVIDIA GeForce GTX 1080"
        ));
        assert!(!CudaArchitecture::has_tensor_cores(
            52,
            "NVIDIA GeForce GTX 980"
        ));
    }

    /// A driver that will not name the device leaves the old behaviour rather than guessing a
    /// card has no tensor cores, which would silently disable CMMA on hardware that has them.
    #[test]
    fn an_unnamed_device_keeps_what_its_version_says() {
        assert!(CudaArchitecture::has_tensor_cores(
            75,
            "unknown CUDA device"
        ));
    }
}

impl Display for CudaArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.version)
    }
}

impl Architecture for CudaArchitecture {
    fn warp_size(&self) -> u32 {
        32
    }

    fn is_wmma_capable(&self) -> bool {
        self.tensor_cores
    }

    fn is_mfma_capable(&self) -> bool {
        false
    }

    fn get_version(&self) -> u32 {
        self.version
    }
}
