//! What an AMD gfx architecture is, for every backend that generates code for one.
//!
//! The HIP runtime drives two of them, and the wavefront width and WMMA generation a
//! kernel is built for have to be the same answer whichever one compiles it.

use alloc::string::String;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl AMDArchitecture {
    /// `None` for architectures without WMMA at all.
    pub fn wmma_generation(&self) -> Option<AmdWmma> {
        match self {
            AMDArchitecture::GFX11 => Some(AmdWmma::Rdna3),
            AMDArchitecture::GFX12 => Some(AmdWmma::Rdna4),
            _ => None,
        }
    }

    /// The wavefront width, or `None` for an architecture this table does not know.
    ///
    /// CDNA runs wave64 (gfx9 and gfx940+) and RDNA wave32 (gfx10, gfx11, gfx12).
    pub fn plane_dim(&self) -> Option<u32> {
        match self {
            AMDArchitecture::GFX10 | AMDArchitecture::GFX11 | AMDArchitecture::GFX12 => Some(32),
            AMDArchitecture::GFX908 | AMDArchitecture::GFX90A | AMDArchitecture::GFX94 => Some(64),
            AMDArchitecture::Other => None,
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

/// A gfx architecture name, parsed once from what the driver reports.
///
/// `gcnArchName` carries a target-feature suffix (`gfx1151:xnack-`), and the bare name in
/// front of it is what selects a `-mcpu`, a device library and a fragment layout. Both AMD
/// backends ask this the same questions, so they get the same answers: the wavefront width
/// the compiler generates for is the one the runtime checks against the driver.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GfxArch {
    name: String,
    family: AMDArchitecture,
}

impl GfxArch {
    /// Parse `gcnArchName` as the driver reports it, suffix included.
    pub fn parse(reported: &str) -> Self {
        let name = reported
            .split(':')
            .next()
            .unwrap_or(reported)
            .to_lowercase();
        let family = AMDArchitecture::parse(&name).unwrap_or(AMDArchitecture::Other);
        Self { name, family }
    }

    /// The bare name, which is the `-mcpu` and the `target-cpu` attribute.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Which family the tables key off.
    pub fn family(&self) -> AMDArchitecture {
        self.family
    }

    /// The wavefront width, or `None` for an architecture the table does not know.
    pub fn plane_dim(&self) -> Option<u32> {
        self.family.plane_dim()
    }

    /// Which WMMA this device has, or `None` where it has none at all.
    pub fn wmma(&self) -> Option<AmdWmma> {
        self.family.wmma_generation()
    }

    /// The bare architecture number, which is how the `oclc_isa_version_*` control libraries
    /// are named.
    pub fn isa_version(&self) -> &str {
        self.name.strip_prefix("gfx").unwrap_or(&self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    /// The driver appends target features to the architecture name, and everything keyed on
    /// the architecture wants the bare one in front of them.
    #[test]
    fn the_target_feature_suffix_is_not_part_of_the_name() {
        let gfx = GfxArch::parse("gfx1151:xnack-:sramecc+");
        assert_eq!(gfx.name(), "gfx1151");
        assert_eq!(gfx.isa_version(), "1151");
        assert_eq!(gfx.family(), AMDArchitecture::GFX11);
    }

    /// The wavefront width and the WMMA generation are one answer per device, given to both
    /// backends. A disagreement means kernels are generated for the wrong wave width.
    #[test]
    fn a_device_has_one_wave_width_and_one_wmma() {
        for (name, plane_dim, wmma) in [
            ("gfx1201", Some(32), Some(AmdWmma::Rdna4)),
            ("gfx1100", Some(32), Some(AmdWmma::Rdna3)),
            ("gfx1030", Some(32), None),
            ("gfx90a", Some(64), None),
            ("gfx942", Some(64), None),
        ] {
            let gfx = GfxArch::parse(name);
            assert_eq!(gfx.plane_dim(), plane_dim, "{name}");
            assert_eq!(gfx.wmma(), wmma, "{name}");
        }
    }

    /// An architecture the table has never heard of reports no width rather than guessing
    /// one, so the runtime refuses it instead of compiling every kernel for the wrong wave.
    #[test]
    fn an_unknown_architecture_claims_no_wave_width() {
        let gfx = GfxArch::parse("gfx1337");
        assert_eq!(gfx.family(), AMDArchitecture::Other);
        assert_eq!(gfx.plane_dim(), None);
        assert_eq!(gfx.name().to_string(), "gfx1337");
    }
}
