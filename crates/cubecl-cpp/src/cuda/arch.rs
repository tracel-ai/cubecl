use std::str::FromStr;

use crate::shared::Architecture;

pub struct CudaArchitecture {
    pub version: u32,
}

impl FromStr for CudaArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let version = s.parse().map_err(|e| format!("bad cuda architecture: {e}"))?;
        Ok(Self { version })
    }
}

impl Architecture for CudaArchitecture {
    fn warp_size(&self) -> u32 {
        32
    }

    fn is_wmma_capable(&self) -> bool {
        true
    }
}
