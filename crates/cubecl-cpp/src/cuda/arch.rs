use std::{fmt::Display, str::FromStr};

use crate::shared::Architecture;

#[derive(Debug)]
pub struct CudaArchitecture {
    pub version: u32,
}

impl FromStr for CudaArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let version = s
            .parse()
            .map_err(|e| format!("bad cuda architecture: {e}"))?;
        Ok(Self { version })
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
        true
    }

    fn is_mfma_capable(&self) -> bool {
        false
    }

    fn get_version(&self) -> u32 {
        self.version
    }
}
