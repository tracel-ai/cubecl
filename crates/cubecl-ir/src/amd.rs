//! What an AMD gfx architecture is, for every backend that generates code for one.
//!
//! The HIP runtime drives two of them, and the wavefront width and WMMA generation a
//! kernel is built for have to be the same answer whichever one compiles it.

use crate::{
    ContiguousElements, CoordinatePart, ElementIndex, ElementPositions, MmaProperties, OperandFact,
    interfaces::TypedExt,
    types::matrix::{MatrixIdent, MatrixLayout, MatrixShape, MatrixType},
};
use alloc::{string::String, vec, vec::Vec};
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
    pub fn frag_ab_elems_at_depth(&self, k: usize) -> usize {
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

/// The lanes whose places along one side of a tile differ, the next ones starting over.
const ROW_LANE_COUNT: usize = 16;

/// The lane groups an MFMA instruction splits `k` between: a 64-wide wavefront of 16-lane rows.
pub const MFMA_LANE_GROUP_COUNT: usize = 4;

/// The matrix instructions an AMD part has: RDNA's WMMA or CDNA's MFMA.
///
/// # Assumptions
///
/// - A tile is 16 cells wide on the side a lane names a place of: the lanes count their place
///   over 16.
///
/// # Guarantees
///
/// - Every answer is the same whichever backend lowers the instruction, so the fragments one
///   backend builds are the ones another reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AmdMatrixCore {
    Wmma(AmdWmma),
    Mfma(AmdMfma),
}

impl AmdMatrixCore {
    /// Elements of A/B each lane holds for a tile `k` deep. MFMA splits `k` between four lane
    /// groups, each holding its own run of it.
    pub fn frag_ab_elems_at_depth(&self, k: usize) -> usize {
        match self {
            AmdMatrixCore::Wmma(generation) => generation.frag_ab_elems_at_depth(k),
            AmdMatrixCore::Mfma(_) => k / MFMA_LANE_GROUP_COUNT,
        }
    }

    /// How the manual MMA lays its registers out on this core: `A` along its rows, `B` along its
    /// columns. RDNA3 holds every A/B cell in both halves of its 32-lane wavefront, RDNA4 once;
    /// CDNA splits `k` between the four 16-lane groups of its 64-lane wavefront.
    ///
    /// The accumulator lies along its columns, except on RDNA4: there A and B hold their
    /// registers alike, so the backends run the instruction with them swapped and get the
    /// transposed accumulator, along its rows as A is. A product then feeds the next one as its
    /// A in registers, and each lane holds whole rows for a factor per row.
    ///
    /// # Guarantees
    ///
    /// - Each `register_layout_*` is the [`run_layout`](Self::run_layout) of its operand.
    pub fn mma_properties(self) -> MmaProperties {
        let (plane_lanes, copies_ab) = match self {
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
            const_plane_lane_count: plane_lanes,
            register_layout_a: self.run_layout(MatrixIdent::A),
            register_layout_b: self.run_layout(MatrixIdent::B),
            register_layout_acc: self.run_layout(MatrixIdent::Accumulator),
            register_copy_count_a: copies_ab,
            register_copy_count_b: copies_ab,
            contiguous_elements,
            element_positions: OperandFact::new(move |ctx, ident, matrix| {
                Some(self.element_positions(ident, matrix.deref(ctx).shape))
            }),
            ..MmaProperties::default()
        }
    }

    /// Where the elements a lane holds of operand `ident` of a tile of `shape` lie in it.
    ///
    /// # Assumptions
    ///
    /// - `shape` is 16 by 16, and on RDNA4 and CDNA a lane's share of `k` divides `k`.
    ///
    /// # Guarantees
    ///
    /// - The lanes of a plane together hold every cell of the operand, each as many times as
    ///   [`mma_properties`](Self::mma_properties) says it is copied.
    /// - A lane's place counts along the side [`lane_names_row`](Self::lane_names_row) names,
    ///   and its run of elements along the other.
    pub fn element_positions(self, ident: MatrixIdent, shape: MatrixShape) -> ElementPositions {
        let across = CoordinatePart::of(ElementIndex::Lane).counted_over(ROW_LANE_COUNT);
        let along = self.run_along(ident, shape);
        match self.lane_names_row(ident) {
            true => ElementPositions {
                row: vec![across],
                col: along,
            },
            false => ElementPositions {
                row: along,
                col: vec![across],
            },
        }
    }

    /// Whether a lane's place in its row of 16 lanes names the row of the matrix it holds, its run
    /// of elements then lying along that row; otherwise it names a column.
    ///
    /// # Guarantees
    ///
    /// - A lane names a row of A and a column of B, on every core.
    /// - A lane names a row of the accumulator only on RDNA4, which runs the instruction
    ///   transposed and holds the accumulator as it holds A.
    pub fn lane_names_row(self, ident: MatrixIdent) -> bool {
        match ident {
            MatrixIdent::A => true,
            MatrixIdent::B => false,
            MatrixIdent::Accumulator => self == AmdMatrixCore::Wmma(AmdWmma::Rdna4),
        }
    }

    /// The layout of memory in which a lane's run of elements of operand `ident` is contiguous.
    ///
    /// # Guarantees
    ///
    /// - `RowMajor` where the lane names a row, its run lying along that row, and `ColMajor`
    ///   where it names a column.
    pub fn run_layout(self, ident: MatrixIdent) -> MatrixLayout {
        match self.lane_names_row(ident) {
            true => MatrixLayout::RowMajor,
            false => MatrixLayout::ColMajor,
        }
    }
}

impl AmdMatrixCore {
    /// Where a lane's run of elements lies along the side of a tile it does not name a place of.
    fn run_along(self, ident: MatrixIdent, shape: MatrixShape) -> Vec<CoordinatePart> {
        let nth = CoordinatePart::of(ElementIndex::Nth);
        let group = CoordinatePart::of(ElementIndex::Lane).in_runs_of(ROW_LANE_COUNT);
        match (ident, self) {
            // Both halves of an RDNA3 wavefront hold the whole of `k`.
            (MatrixIdent::A | MatrixIdent::B, AmdMatrixCore::Wmma(AmdWmma::Rdna3)) => vec![nth],
            (MatrixIdent::A | MatrixIdent::B, _) => {
                vec![nth, group.spaced_by(self.frag_ab_elems_at_depth(shape.k))]
            }
            (MatrixIdent::Accumulator, AmdMatrixCore::Wmma(AmdWmma::Rdna3)) => {
                vec![nth.spaced_by(2), group]
            }
            (MatrixIdent::Accumulator, AmdMatrixCore::Wmma(AmdWmma::Rdna4)) => {
                vec![nth, group.spaced_by(shape.n / 2)]
            }
            (MatrixIdent::Accumulator, AmdMatrixCore::Mfma(_)) => {
                vec![nth, group.spaced_by(shape.m / MFMA_LANE_GROUP_COUNT)]
            }
        }
    }

    /// Consecutive elements of `matrix` a lane holds in role `ident`: for A and B a lane's run of
    /// `k`, at most a load wide, and for the accumulator the half of its row RDNA4 gives a lane,
    /// whole, so a lane keeps one state for the one row it holds, or else one element at a time.
    fn contiguous_elements(
        self,
        ctx: &Context,
        ident: MatrixIdent,
        matrix: TypedHandle<MatrixType>,
    ) -> usize {
        let matrix = matrix.deref(ctx);
        let widest = LOAD_WIDTH_BYTES / matrix.elem_ty.size(ctx);
        match ident {
            MatrixIdent::A | MatrixIdent::B => {
                self.frag_ab_elems_at_depth(matrix.shape.k).min(widest)
            }
            MatrixIdent::Accumulator => match self {
                AmdMatrixCore::Wmma(AmdWmma::Rdna4) => matrix.shape.n / 2,
                AmdMatrixCore::Wmma(AmdWmma::Rdna3) | AmdMatrixCore::Mfma(_) => 1,
            },
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

    /// The family of a bare gfx architecture name, `Other` for one this table does not know.
    pub fn parse(arg: &str) -> Self {
        match arg.to_lowercase().as_str() {
            norm if norm.starts_with("gfx120") => AMDArchitecture::GFX12,
            norm if norm.starts_with("gfx11") => AMDArchitecture::GFX11,
            norm if norm.starts_with("gfx103") => AMDArchitecture::GFX103,
            norm if norm.starts_with("gfx101") => AMDArchitecture::GFX101,
            "gfx908" => AMDArchitecture::GFX908,
            "gfx90a" => AMDArchitecture::GFX90A,
            "gfx900" | "gfx902" | "gfx904" | "gfx906" | "gfx909" | "gfx90c" => {
                AMDArchitecture::GFX9
            }
            norm if norm.starts_with("gfx94") => AMDArchitecture::GFX94,
            _ => AMDArchitecture::Other,
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
        let family = AMDArchitecture::parse(&name);
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

    /// Every cell of every operand is held, as many times as the properties say it is.
    #[test]
    fn the_lanes_of_a_plane_hold_an_operand_together() {
        for core in [
            AmdMatrixCore::Wmma(AmdWmma::Rdna3),
            AmdMatrixCore::Wmma(AmdWmma::Rdna4),
            AmdMatrixCore::Mfma(AmdMfma::Cdna3),
        ] {
            let mma = core.mma_properties();
            for (ident, duplication) in [
                (MatrixIdent::A, mma.register_copy_count_a),
                (MatrixIdent::B, mma.register_copy_count_b),
                (MatrixIdent::Accumulator, mma.register_copy_count_acc),
            ] {
                let shape = MatrixShape::from((16, 16, 32));
                let cells = shape.num_elems(ident);
                let lanes = mma.const_plane_lane_count as usize;
                let held = core.element_positions(ident, shape).held_cells(
                    (shape, ident),
                    lanes,
                    cells * duplication / lanes,
                );
                assert_eq!(held, vec![duplication; cells], "{core:?} {ident}");
            }
        }
    }

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
    }

    /// MFMA splits `k` between its four lane groups.
    #[test]
    fn an_mfma_lane_holds_a_quarter_of_k() {
        let cdna3 = AmdMatrixCore::Mfma(AmdMfma::Cdna3);
        assert_eq!(cdna3.frag_ab_elems_at_depth(16), 4);
        assert_eq!(cdna3.frag_ab_elems_at_depth(32), 8);
    }

    /// A lane's run lies along the side it names: A's rows, B's columns, and the accumulator's
    /// columns except on RDNA4.
    #[test]
    fn a_lane_run_is_contiguous_along_the_side_it_names() {
        use MatrixIdent::{A, Accumulator, B};
        use MatrixLayout::{ColMajor, RowMajor};

        for core in [
            AmdMatrixCore::Wmma(AmdWmma::Rdna3),
            AmdMatrixCore::Mfma(AmdMfma::Cdna3),
        ] {
            assert_eq!(core.run_layout(A), RowMajor, "{core:?}");
            assert_eq!(core.run_layout(B), ColMajor, "{core:?}");
            assert_eq!(core.run_layout(Accumulator), ColMajor, "{core:?}");
        }
    }

    /// RDNA4 runs the instruction transposed, so its accumulator is held along its rows.
    #[test]
    fn an_rdna4_accumulator_run_is_contiguous_along_its_rows() {
        let rdna4 = AmdMatrixCore::Wmma(AmdWmma::Rdna4);
        assert_eq!(
            rdna4.run_layout(MatrixIdent::Accumulator),
            MatrixLayout::RowMajor
        );
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
