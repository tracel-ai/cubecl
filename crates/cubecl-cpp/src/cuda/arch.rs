use std::fmt::Display;

use crate::shared::Architecture;

#[derive(Debug)]
pub struct CudaArchitecture {
    pub version: u32,
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
        true
    }

    fn is_mfma_capable(&self) -> bool {
        false
    }

    fn get_version(&self) -> u32 {
        self.version
    }
}
