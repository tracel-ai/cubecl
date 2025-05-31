use std::fmt::Display;

use crate::shared::Architecture;

// We support Metal 3 family of GPUs

pub enum MetalArchitecture {
    Metal3,
    Other,
}

impl Display for MetalArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Metal3 => write!(f, "metal3"),
            Self::Other => write!(f, "other"),
        }
    }
}

impl MetalArchitecture {
    pub fn parse(arg: &str) -> Result<Self, String> {
        let norm = arg.to_lowercase();
        if norm.starts_with("metal3") {
            Ok(MetalArchitecture::Metal3)
        } else {
            Ok(MetalArchitecture::Other)
        }
    }
}

impl Architecture for MetalArchitecture {
    fn warp_size(&self) -> u32 {
        64
    }

    fn is_wmma_capable(&self) -> bool {
        true
    }

    fn is_mfma_capable(&self) -> bool {
        false
    }
}
