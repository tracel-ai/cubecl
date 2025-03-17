use std::{fmt::Display, str::FromStr};

use crate::shared::Architecture;

// We support Metal 3 family of GPUs

pub enum MetalArchitecture {
    Metal3,
    Other,
}

impl FromStr for MetalArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let norm = s.to_lowercase();
        if norm.starts_with("metal3") {
            Ok(MetalArchitecture::Metal3)
        } else {
            Ok(MetalArchitecture::Other)
        }
    }
}

impl Display for MetalArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Metal3 => write!(f, "metal3"),
            Self::Other => write!(f, "other"),
        }
    }
}

impl Architecture for MetalArchitecture {
    fn warp_size(&self) -> u32 {
        32
    }

    fn is_wmma_capable(&self) -> bool {
        false
    }

    fn is_mfma_capable(&self) -> bool {
        false
    }
}
