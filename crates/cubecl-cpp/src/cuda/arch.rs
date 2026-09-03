use std::fmt::Display;

use crate::shared::Architecture;

#[derive(Debug)]
pub struct CudaArchitecture {
    pub version: u32,
    /// Compute capability cannot say this: TU116 and TU117 report 7.5 like every other
    /// Turing, and `mma.sync` still runs on them, on the FP16 pipeline at a fiftieth the rate.
    pub tensor_cores: bool,
}

impl CudaArchitecture {
    /// Decided by the marketing name because CUDA exposes no attribute for it. No GTX-branded
    /// Turing has tensor cores and every Turing that does shipped under another brand, so the
    /// two coincide exactly at 7.5. The professional TU117 parts (T400 to T1000) are not caught.
    pub fn has_tensor_cores(version: u32, name: &str) -> bool {
        match version {
            // Tensor cores arrive with Volta.
            ..70 => false,
            // The GTX-branded Turings are the only later dies without them.
            75 => !name.to_uppercase().contains("GTX"),
            _ => true,
        }
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

#[cfg(test)]
mod tests {
    use super::CudaArchitecture;

    #[test]
    fn turing_without_tensor_cores_is_told_apart_from_turing_with_them() {
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
        assert!(CudaArchitecture::has_tensor_cores(
            80,
            "NVIDIA A100-GTX-ish"
        ));
        // Tesla T4 is TU104: 7.5 with tensor cores, no GTX in the name.
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
        // Volta itself has them: the boundary is inclusive.
        assert!(CudaArchitecture::has_tensor_cores(70, "Tesla V100-SXM2"));
    }

    #[test]
    fn an_unnamed_device_keeps_what_its_version_says() {
        assert!(CudaArchitecture::has_tensor_cores(
            75,
            "unknown CUDA device"
        ));
    }
}
