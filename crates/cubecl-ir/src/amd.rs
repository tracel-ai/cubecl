//! What an AMD gfx architecture is, for every backend that generates code for one.
//!
//! The HIP runtime drives two of them, and the wavefront width and WMMA generation a
//! kernel is built for have to be the same answer whichever one compiles it.

use crate::{
    ContiguousElements, MmaProperties,
    interfaces::TypedExt,
    types::matrix::{MatrixIdent, MatrixLayout, MatrixType},
};
use alloc::string::String;
use pliron::{context::Context, r#type::TypedHandle};

/// The bytes a lane loads of an operand in one go: a swizzle atom and the widest load.
const LOAD_WIDTH_BYTES: usize = 16;

/// The bits each register of a matrix fragment holds.
const REGISTER_BITS: usize = 32;

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

/// Which generation of CDNA's MFMA instructions to emit. Both lay out the 16x16 forms alike,
/// over four groups of 16 lanes of a 64-wide wavefront, but CDNA3 replaced CDNA2's 16-deep
/// 8-bit integer form with a 32-deep one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmdMfma {
    /// gfx90a (MI200).
    Cdna2,
    /// gfx940, gfx941, gfx942 (MI300).
    Cdna3,
}

/// The lane groups an MFMA instruction splits `k` between: a 64-wide wavefront of 16-lane rows.
pub const MFMA_LANE_GROUPS: usize = 4;

/// The matrix instructions an AMD part has: RDNA's WMMA or CDNA's MFMA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmdMatrixCore {
    Wmma(AmdWmma),
    Mfma(AmdMfma),
}

impl AmdMatrixCore {
    /// Elements of A/B each lane holds for a tile `k` deep. MFMA splits `k` between four lane
    /// groups, each holding its own run of it.
    pub fn frag_ab_elems(&self, k: usize) -> usize {
        match self {
            AmdMatrixCore::Wmma(generation) => generation.frag_ab_elems(k),
            AmdMatrixCore::Mfma(_) => k / MFMA_LANE_GROUPS,
        }
    }

    /// How the manual MMA lays its registers out on this core, whichever backend lowers it: `A`
    /// along its rows, `B` and the accumulator along their columns. RDNA3 holds every A/B cell
    /// in both halves of its 32-lane wavefront, RDNA4 once; CDNA splits `k` between the four
    /// 16-lane groups of its 64-lane wavefront.
    pub fn mma_properties(self) -> MmaProperties {
        let (plane_size, duplication_ab) = match self {
            AmdMatrixCore::Wmma(AmdWmma::Rdna3) => (32, 2),
            AmdMatrixCore::Wmma(AmdWmma::Rdna4) => (32, 1),
            AmdMatrixCore::Mfma(_) => (64, 1),
        };
        let contiguous_elements = match self {
            AmdMatrixCore::Wmma(AmdWmma::Rdna3) => ContiguousElements::new(contiguous_rdna3),
            AmdMatrixCore::Wmma(AmdWmma::Rdna4) => ContiguousElements::new(contiguous_rdna4),
            AmdMatrixCore::Mfma(_) => ContiguousElements::new(contiguous_cdna),
        };
        MmaProperties {
            register_size_bits: REGISTER_BITS,
            const_plane_size: plane_size,
            register_layout_a: MatrixLayout::RowMajor,
            register_layout_b: MatrixLayout::ColMajor,
            register_layout_acc: MatrixLayout::ColMajor,
            register_duplication_a: duplication_ab,
            register_duplication_b: duplication_ab,
            register_duplication_acc: 1,
            contiguous_elements,
        }
    }

    /// Consecutive elements of `matrix` a lane holds in role `ident`, at most a load wide: a
    /// lane's run of `k` for A and B, one accumulator element at a time.
    fn contiguous_elements(
        self,
        ctx: &Context,
        ident: MatrixIdent,
        matrix: TypedHandle<MatrixType>,
    ) -> usize {
        let matrix = matrix.deref(ctx);
        let widest = LOAD_WIDTH_BYTES / matrix.elem_ty.size(ctx);
        match ident {
            MatrixIdent::A | MatrixIdent::B => self.frag_ab_elems(matrix.shape.k).min(widest),
            MatrixIdent::Accumulator => 1,
        }
    }
}

fn contiguous_rdna3(ctx: &Context, ident: MatrixIdent, matrix: TypedHandle<MatrixType>) -> usize {
    AmdMatrixCore::Wmma(AmdWmma::Rdna3).contiguous_elements(ctx, ident, matrix)
}

fn contiguous_rdna4(ctx: &Context, ident: MatrixIdent, matrix: TypedHandle<MatrixType>) -> usize {
    AmdMatrixCore::Wmma(AmdWmma::Rdna4).contiguous_elements(ctx, ident, matrix)
}

/// Both CDNA generations lay a lane's run of `k` out alike.
fn contiguous_cdna(ctx: &Context, ident: MatrixIdent, matrix: TypedHandle<MatrixType>) -> usize {
    AmdMatrixCore::Mfma(AmdMfma::Cdna3).contiguous_elements(ctx, ident, matrix)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AMDArchitecture {
    // RDNA
    // gfx1200, gfx1201 (RDNA4). gfx1250 is no RDNA4: its WMMA forms differ.
    GFX12,
    // gfx1100, gfx1101, gfx1102
    GFX11,
    // gfx1030 through gfx1036 (RDNA2)
    GFX103,
    // gfx1010 through gfx1013 (RDNA1): wave32, but no WMMA, which arrives with RDNA3
    GFX101,
    // GCN5 (Vega)
    // gfx900, gfx902, gfx904, gfx906, gfx909, gfx90c: wave64, with neither WMMA nor MFMA
    GFX9,
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

    /// `None` for architectures without MFMA, or with one this table does not lower: CDNA1
    /// (gfx908) has no 16x16 `bf16` form taking four elements a lane.
    pub fn mfma_generation(&self) -> Option<AmdMfma> {
        match self {
            AMDArchitecture::GFX90A => Some(AmdMfma::Cdna2),
            AMDArchitecture::GFX94 => Some(AmdMfma::Cdna3),
            _ => None,
        }
    }

    /// The matrix instructions this architecture has, `None` where it has neither kind.
    pub fn matrix_core(&self) -> Option<AmdMatrixCore> {
        self.wmma_generation()
            .map(AmdMatrixCore::Wmma)
            .or_else(|| self.mfma_generation().map(AmdMatrixCore::Mfma))
    }

    /// The wavefront width, or `None` for an architecture this table does not know.
    ///
    /// CDNA runs wave64 (gfx9 and gfx940+) and RDNA wave32 (gfx10, gfx11, gfx12).
    pub fn plane_dim(&self) -> Option<u32> {
        match self {
            AMDArchitecture::GFX101
            | AMDArchitecture::GFX103
            | AMDArchitecture::GFX11
            | AMDArchitecture::GFX12 => Some(32),
            AMDArchitecture::GFX9
            | AMDArchitecture::GFX908
            | AMDArchitecture::GFX90A
            | AMDArchitecture::GFX94 => Some(64),
            AMDArchitecture::Other => None,
        }
    }

    pub fn parse(arg: &str) -> Result<Self, String> {
        let norm = arg.to_lowercase();
        if norm.starts_with("gfx120") {
            Ok(AMDArchitecture::GFX12)
        } else if norm.starts_with("gfx11") {
            Ok(AMDArchitecture::GFX11)
        } else if norm.starts_with("gfx103") {
            Ok(AMDArchitecture::GFX103)
        } else if norm.starts_with("gfx101") {
            Ok(AMDArchitecture::GFX101)
        } else if norm == "gfx908" {
            Ok(AMDArchitecture::GFX908)
        } else if norm == "gfx90a" {
            Ok(AMDArchitecture::GFX90A)
        } else if matches!(
            norm.as_str(),
            "gfx900" | "gfx902" | "gfx904" | "gfx906" | "gfx909" | "gfx90c"
        ) {
            Ok(AMDArchitecture::GFX9)
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

    /// Which matrix instructions this device has, or `None` where it has neither kind.
    pub fn matrix_core(&self) -> Option<AmdMatrixCore> {
        self.family.matrix_core()
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
            ("gfx1250", None, None),
            ("gfx1100", Some(32), Some(AmdWmma::Rdna3)),
            ("gfx1030", Some(32), None),
            ("gfx1010", Some(32), None),
            ("gfx90c", Some(64), None),
            ("gfx90a", Some(64), None),
            ("gfx942", Some(64), None),
        ] {
            let gfx = GfxArch::parse(name);
            assert_eq!(gfx.plane_dim(), plane_dim, "{name}");
            assert_eq!(gfx.wmma(), wmma, "{name}");
        }
    }

    #[test]
    fn rdna1_is_not_rdna2() {
        assert_eq!(GfxArch::parse("gfx1010").family(), AMDArchitecture::GFX101);
        assert_eq!(GfxArch::parse("gfx1012").family(), AMDArchitecture::GFX101);
        assert_eq!(GfxArch::parse("gfx1030").family(), AMDArchitecture::GFX103);
        assert_eq!(GfxArch::parse("gfx1032").family(), AMDArchitecture::GFX103);
        assert_eq!(GfxArch::parse("gfx1010").plane_dim(), Some(32));
        assert_eq!(GfxArch::parse("gfx1030").plane_dim(), Some(32));
        assert_eq!(GfxArch::parse("gfx1010").wmma(), None);
    }

    /// GCN5 shares the `gfx90` prefix with CDNA1 and CDNA2, which have MFMA.
    #[test]
    fn gcn5_is_not_cdna() {
        for name in [
            "gfx900",
            "gfx902",
            "gfx904",
            "gfx906",
            "gfx909",
            "gfx90c:xnack+",
        ] {
            assert_eq!(
                GfxArch::parse(name).family(),
                AMDArchitecture::GFX9,
                "{name}"
            );
        }
        assert_eq!(GfxArch::parse("gfx908").family(), AMDArchitecture::GFX908);
        assert_eq!(GfxArch::parse("gfx90a").family(), AMDArchitecture::GFX90A);
    }

    /// MI300 and MI200 multiply on MFMA, RDNA on WMMA, and nothing else on either.
    #[test]
    fn each_part_has_one_kind_of_matrix_core() {
        for (name, core) in [
            ("gfx942", Some(AmdMatrixCore::Mfma(AmdMfma::Cdna3))),
            ("gfx940", Some(AmdMatrixCore::Mfma(AmdMfma::Cdna3))),
            ("gfx90a", Some(AmdMatrixCore::Mfma(AmdMfma::Cdna2))),
            ("gfx908", None),
            ("gfx1100", Some(AmdMatrixCore::Wmma(AmdWmma::Rdna3))),
            ("gfx1201", Some(AmdMatrixCore::Wmma(AmdWmma::Rdna4))),
            ("gfx1030", None),
        ] {
            assert_eq!(GfxArch::parse(name).matrix_core(), core, "{name}");
        }
        let cdna3 = AmdMatrixCore::Mfma(AmdMfma::Cdna3);
        assert_eq!(cdna3.frag_ab_elems(16), 4);
        assert_eq!(cdna3.frag_ab_elems(32), 8);
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
