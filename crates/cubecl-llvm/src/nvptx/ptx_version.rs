//! The PTX ISA version emitted for the installed driver.

use core::fmt;
use std::ffi::CString;

/// A PTX ISA version, as the NVPTX target feature spells it: 8.7 is `87`.
///
/// Chosen from the driver rather than left to LLVM, whose default is the oldest the architecture
/// accepts: on Turing that predates the `mma.m16n8k8` and `ldmatrix` the runtime advertises
/// there. NVRTC writes its toolkit's version instead, which an older driver can refuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PtxVersion(u32);

impl PtxVersion {
    /// The newest version a driver of CUDA `driver` loads, as `cuDriverGetVersion` reports it
    /// (12.8 is `12080`). `None` below CUDA 10.0, which leaves LLVM's default. It is never below
    /// the architecture's own minimum: a driver that old cannot run the device at all.
    pub fn for_driver(driver: i32) -> Option<Self> {
        DRIVER_PTX
            .iter()
            .find(|(first_driver, _)| driver >= *first_driver)
            .map(|(_, ptx)| Self(*ptx))
    }

    /// The NVPTX target feature that selects this version.
    pub fn target_feature(self) -> CString {
        CString::new(format!("+{self}")).expect("the feature has no NUL")
    }
}

impl fmt::Display for PtxVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ptx{}", self.0)
    }
}

/// The first CUDA release to load each PTX version, newest first, as clang maps them. It stops
/// at 9.3, the newest this LLVM emits: a newer driver still loads it.
const DRIVER_PTX: &[(i32, u32)] = &[
    (13030, 93),
    (13020, 92),
    (13010, 91),
    (13000, 90),
    (12090, 88),
    (12080, 87),
    (12050, 85),
    (12040, 84),
    (12030, 83),
    (12020, 82),
    (12010, 81),
    (12000, 80),
    (11080, 78),
    (11070, 77),
    (11060, 76),
    (11050, 75),
    (11040, 74),
    (11030, 73),
    (11020, 72),
    (11010, 71),
    (11000, 70),
    (10020, 65),
    (10010, 64),
    (10000, 63),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_driver_gets_the_newest_version_it_loads() {
        assert_eq!(PtxVersion::for_driver(12080), Some(PtxVersion(87)));
        // 12.5 and 12.6 both stop at 8.5.
        assert_eq!(PtxVersion::for_driver(12060), Some(PtxVersion(85)));
        assert_eq!(PtxVersion::for_driver(13050), Some(PtxVersion(93)));
        assert_eq!(PtxVersion::for_driver(8000), None);
    }

    #[test]
    fn the_target_feature_names_the_version() {
        assert_eq!(PtxVersion(87).target_feature().as_c_str(), c"+ptx87");
    }
}
