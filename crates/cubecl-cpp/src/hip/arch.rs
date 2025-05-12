use crate::shared::Architecture;

pub enum AMDArchitecture {
    // RDNA
    // gfx1100, gfx1101, gfx1102
    GFX11,
    // gfx1030, gfx1031, gfx1032
    GFX10,
    // CDNA
    GFX908,
    GFX90A,
    // gfx940, gfx941, gfx942
    GFX94,
    // Not particularly specific architecture
    Other,
}

impl AMDArchitecture {
    pub fn parse(arg: &str) -> Result<Self, String> {
        let norm = arg.to_lowercase();
        if norm.starts_with("gfx11") {
            Ok(AMDArchitecture::GFX11)
        } else if norm.starts_with("gfx10") {
            Ok(AMDArchitecture::GFX10)
        } else if norm == "gfx908" {
            Ok(AMDArchitecture::GFX908)
        } else if norm == "gfx90a" {
            Ok(AMDArchitecture::GFX90A)
        } else if norm.starts_with("gfx94") {
            Ok(AMDArchitecture::GFX94)
        } else {
            Ok(AMDArchitecture::Other)
        }
    }
}

impl Architecture for AMDArchitecture {
    fn warp_size(&self) -> u32 {
        // CDNA supports wave64 (gfx9 and gfx940+) and RDNA wave32 (gfx10, gfx11)
        match self {
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 => 32,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94 => 64,
            AMDArchitecture::Other => 0,
        }
    }

    fn is_wmma_capable(&self) -> bool {
        matches!(self, AMDArchitecture::GFX10 | AMDArchitecture::GFX11)
    }

    fn is_mfma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94
        )
    }
}
