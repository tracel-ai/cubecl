use crate::shared::Architecture;

pub enum AMDArchitecture {
    // RDNA
    // gfx1200, gfx1201 (RDNA4)
    GFX12,
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

/// Which generation of AMD's WMMA instructions to emit. RDNA4 renamed the builtins and dropped
/// RDNA3's duplication of the A/B fragments across lane halves, halving them to 8 elements, so the
/// two aren't interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmdWmma {
    Rdna3,
    Rdna4,
}

impl AmdWmma {
    /// Elements of A/B each lane holds. RDNA3 gives every lane the whole `k` range and duplicates
    /// it across lanes 0-15 / 16-31; RDNA4 splits `k` between the two halves instead.
    pub fn frag_ab_elems(&self, k: usize) -> usize {
        match self {
            AmdWmma::Rdna3 => k,
            AmdWmma::Rdna4 => k / 2,
        }
    }
}

impl AMDArchitecture {
    /// `None` for architectures without WMMA at all.
    pub fn wmma_generation(&self) -> Option<AmdWmma> {
        match self {
            AMDArchitecture::GFX11 => Some(AmdWmma::Rdna3),
            AMDArchitecture::GFX12 => Some(AmdWmma::Rdna4),
            _ => None,
        }
    }

    pub fn parse(arg: &str) -> Result<Self, String> {
        let norm = arg.to_lowercase();
        if norm.starts_with("gfx12") {
            Ok(AMDArchitecture::GFX12)
        } else if norm.starts_with("gfx11") {
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
        // CDNA supports wave64 (gfx9 and gfx940+) and RDNA wave32 (gfx10, gfx11, gfx12)
        match self {
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 | AMDArchitecture::GFX12 => 32,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94 => 64,
            AMDArchitecture::Other => 0,
        }
    }

    fn is_wmma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 | AMDArchitecture::GFX12
        )
    }

    fn is_mfma_capable(&self) -> bool {
        matches!(
            self,
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94
        )
    }
}
