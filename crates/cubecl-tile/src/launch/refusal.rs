//! [`Refusal`]: why a launch cannot run as described, or an operand cannot be bound to it.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::Tiling;

use crate::{Axis, LineMisfit};

/// Why a launch cannot run as described, or an operand cannot be bound to it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// [`in_stride_order`](super::Arg::in_stride_order) on a gathered or storage-tiled operand.
    StrideOrderOfAStatedLayout,
    /// A gathered mapping over a storage-tiled binding.
    GatherOfATiledBinding(Tiling),
    /// [`gathered`](super::Arg::gathered) combined with `axes` or `batches`.
    GatherWithLabels,
    /// The mapping addresses `mapped` dims but the operand has `rank`.
    GatherRankMismatch { mapped: usize, rank: usize },
    /// The mapping spans an axis the launched space does not have.
    GatherOfAnUnknownAxis(Axis),
    /// The operand's rank is smaller than its labelled block of dims.
    RankBelowLabels { rank: usize, block: usize },
    /// More leading dims than batch axes to label them.
    UnlabelledBatchDims { dims: usize, axes: usize },
    /// An operand must address at least one coordinate.
    NoCoordinate,
    /// The buffer cannot be served `width` wide.
    WidthNotServed { width: usize, why: LineMisfit },
    /// The innermost axis is served in vector lines but is not provably in bounds.
    UncheckableVectorEdge,
    /// A gathered mapping read in vector lines whose innermost physical axis does not step by the
    /// operand's last logical axis at coefficient 1.
    GatherInnermostNotInLines,
    /// A gathered mapping that reads `axis` off several physical axes: the operand is
    /// storage-tiled, or reads one axis from two places.
    GatherAxisAddressedTwice(Axis),
    /// A grid read off the levels ([`Grid::FromLevels`](crate::launch::Grid::FromLevels)) needs a
    /// units level of one unit or the whole `plane`; this one holds `units`.
    GridNotReadOffLevels { units: u32, plane: u32 },
    /// A cube of `units` units, past the `most` the device holds.
    CubePastDevice { units: u32, most: u32 },
    /// The kernel works in planes, and the device runs no plane operations.
    PlanesUnavailable,
    /// The kernel needs one plane width, and the device's varies between `min` and `max`.
    ImprecisePlaneDim { min: u32, max: u32 },
    /// A routine declines a problem it does not serve, for the reason `why`.
    Declined { why: &'static str },
    /// A cube shaped `shape` along x, y and z, past the `most` the device spawns along one of them:
    /// a cube legal in units can still be too long on an axis.
    CubeShapePastDevice {
        shape: (u32, u32, u32),
        most: (u32, u32, u32),
    },
    /// A cube with a zero axis, which spawns no units.
    NoUnits,
    /// A grid with a zero axis, which spawns no cubes: the runtime returns without dispatching
    /// anything, and the outputs keep whatever they held.
    NoCubes,
    /// A grid of `count` cubes along x, y and z, past the `most` the device launches along one of
    /// them.
    GridPastDevice {
        count: (u32, u32, u32),
        most: (u32, u32, u32),
    },
    /// A cube holding `bytes` of shared memory, past the `most` the device gives one.
    SharedMemoryPastDevice { bytes: usize, most: usize },
    /// `fillers` planes are set aside to fill a walk's stages, and the device has no barrier type
    /// for the two roles to meet on.
    FillersWithoutBarriers { fillers: u32 },
    /// A TMA box edge past the descriptor's per-axis limit `most`.
    BoxPastDescriptor {
        axis: Axis,
        edge: usize,
        most: usize,
    },
    /// A TMA box lands its rows whole, one after the other, and the stage's blocks split the rows
    /// along `axis`, the innermost, or stack `rows` of them where the swizzle repeats every
    /// `period`.
    BoxSplitsStageBlocks {
        axis: Axis,
        rows: usize,
        period: usize,
    },
    /// A TMA box lands its rows dense, at most swizzled within one span, and the stage keeps them
    /// as `why` says.
    RowsNoDescriptorLands { why: &'static str },
    /// [`paged`](super::Arg::paged) on a gathered or storage-tiled operand: a page is resolved
    /// digit for digit over a plain buffer's labelled dims.
    PagingOfAStatedLayout,
    /// [`paged`](super::Arg::paged) naming an axis no dim of the operand is labelled with.
    PagingOfAnUnlabelledAxis(Axis),
    /// [`paged`](super::Arg::paged) naming one dim for both the pages and their positions.
    PagingOfOneDim(Axis),
    /// [`paged`](super::Arg::paged) with the pages along the innermost dim, this axis's: a line
    /// runs along it, so one line would span pages.
    PagesAlongALine(Axis),
    /// [`paged`](super::Arg::paged) on a pool of no pages, or of pages of no positions: there is
    /// no page for a table entry to read.
    PagingOfAnEmptyPool,
    /// [`paged`](super::Arg::paged) on positions along the innermost dim, read `width` a line,
    /// with pages of `page_size` positions: a line would cross a page.
    PageSplitsALine { page_size: usize, width: usize },
    /// [`paged`](super::Arg::paged) on a packed operand, or one moved by runs (a bulk copy, a
    /// tensor map): the table resolves one line at a time.
    PagingMovedByRuns,
    /// [`indexed`](crate::TileArgLaunch::indexed) with a table that is not plain `u32` entries,
    /// one a line: an entry is a coordinate.
    IndexTableNotU32,
    /// [`indexed`](crate::TileArgLaunch::indexed) with a table of `rank` dims keyed by `keys`
    /// axes: a table holds one dim per key.
    IndexTableRank { keys: usize, rank: usize },
    /// [`indexed`](crate::TileArgLaunch::indexed) on a tensor map, or on a gathered, storage-tiled
    /// or paged operand: an entry moves a window of a plain buffer's labelled dims.
    IndexOfAStatedLayout,
    /// [`indexed`](crate::TileArgLaunch::indexed) placing an axis the operand does not span.
    IndexOfAnAxisNotSpanned(Axis),
    /// [`indexed`](crate::TileArgLaunch::indexed) placing the innermost axis, this one, which the
    /// operand reads more than one value a line along: an entry is an element, and a line would
    /// straddle two of them.
    IndexAlongALine(Axis),
    /// [`indexed`](crate::TileArgLaunch::indexed) with no key, or keyed by the axis it places.
    IndexKeyedByItsAxis(Axis),
    /// [`Indexing::Run`](crate::Indexing::Run) keyed by this many axes: a run's end is the next
    /// entry along its one key.
    RunKeyedByMany(usize),
    /// An indexed operand under a nest whose first level cutting one of these keys does not cut
    /// them all: the table is read at one step.
    IndexKeysCutApart(Vec<Axis>),
    /// An indexed operand under a nest that cuts the axis its entry places: a coordinate is the
    /// table's to state, and a run is placed whole before any level cuts it.
    IndexedAxisCut(Axis),
}

impl Display for Refusal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::StrideOrderOfAStatedLayout => write!(
                f,
                "Arg::in_stride_order: the order a buffer steps is read off strides, which a \
                 gathered mapping states for itself and a storage-tiled binding stores in tiles"
            ),
            Refusal::GatherOfATiledBinding(tiling) => write!(
                f,
                "Arg::gathered: the mapping addresses the buffer's own dims, so a storage-tiled \
                 binding ({tiling:?}) has no reading here"
            ),
            Refusal::GatherWithLabels => write!(
                f,
                "Arg::gathered: the mapping is stated outright, so `axes` and `batches` have \
                 nothing left to describe"
            ),
            Refusal::GatherRankMismatch { mapped, rank } => write!(
                f,
                "Arg::gathered: the mapping addresses {mapped} dims but the operand has {rank}"
            ),
            Refusal::GatherOfAnUnknownAxis(axis) => write!(
                f,
                "Arg::gathered: the mapping spans {axis:?}, which the launched space does not have"
            ),
            Refusal::RankBelowLabels { rank, block } => write!(
                f,
                "Arg: operand rank {rank} is smaller than its labelled block of {block} dims"
            ),
            Refusal::UnlabelledBatchDims { dims, axes } => {
                write!(f, "Arg: {dims} batch dims but only {axes} batch axes given")
            }
            Refusal::NoCoordinate => {
                write!(f, "Arg: an operand must address at least one coordinate")
            }
            Refusal::WidthNotServed { width, why } => {
                write!(
                    f,
                    "Arg::vectorize: this operand cannot be served {width} wide: {why}"
                )
            }
            Refusal::UncheckableVectorEdge => write!(
                f,
                "Arg: the innermost axis is served in vector lines but is not provably in bounds \
                 (it overhangs its tiling, or its map is affine and reaches past any extent stated \
                 here); serve it scalar, or state BoundaryPolicy::Unchecked if the launch proves \
                 its vector lines are in bounds"
            ),
            Refusal::GatherInnermostNotInLines => write!(
                f,
                "Arg::gathered: the innermost physical axis must step by the operand's last \
                 logical axis at coefficient 1 (it is addressed in vector lines)"
            ),
            Refusal::GatherAxisAddressedTwice(axis) => write!(
                f,
                "Arg::gathered: logical axis {axis:?} addresses several physical axes, so it is \
                 either storage-tiled (a gathered operand must be untiled gmem) or read off two \
                 places at once"
            ),
            Refusal::GridNotReadOffLevels { units, plane } => write!(
                f,
                "Launcher: a grid read off the levels needs a units level of one unit or the \
                 whole plane ({plane}), got {units}"
            ),
            Refusal::CubePastDevice { units, most } => write!(
                f,
                "Launcher: a cube of {units} units, but the device holds at most {most}"
            ),
            Refusal::PlanesUnavailable => {
                write!(
                    f,
                    "the kernel works in planes, and the device runs no plane operations"
                )
            }
            Refusal::ImprecisePlaneDim { min, max } => write!(
                f,
                "the kernel needs one plane width, and the device's varies from {min} to {max}"
            ),
            Refusal::Declined { why } => f.write_str(why),
            Refusal::CubeShapePastDevice { shape, most } => write!(
                f,
                "Launcher: a cube shaped {shape:?}, but the device spawns at most {most:?}"
            ),
            Refusal::NoUnits => write!(f, "Launcher: the cube has a zero axis and spawns no units"),
            Refusal::NoCubes => write!(f, "Launcher: the grid has a zero axis and spawns no cubes"),
            Refusal::GridPastDevice { count, most } => write!(
                f,
                "Launcher: a grid of {count:?} cubes, but the device launches at most {most:?}"
            ),
            Refusal::SharedMemoryPastDevice { bytes, most } => write!(
                f,
                "Launcher: a cube holding {bytes} bytes of shared memory, but the device gives \
                 one at most {most}"
            ),
            Refusal::FillersWithoutBarriers { fillers } => write!(
                f,
                "Launcher: {fillers} plane(s) are set aside to fill a walk's stages, and this \
                 device carries no barrier type for the two roles to meet on"
            ),
            Refusal::BoxPastDescriptor { axis, edge, most } => box_past(f, *axis, *edge, *most),
            Refusal::BoxSplitsStageBlocks { axis, rows, period } => write!(
                f,
                "TMA: a box lands its rows whole and in order, and the stage's blocks split \
                 {axis:?} or stack {rows} rows apiece where the swizzle repeats every {period}"
            ),
            Refusal::RowsNoDescriptorLands { why } => write!(
                f,
                "TMA: a box lands its rows dense, at most swizzled within a span, and the stage \
                 holds {why}"
            ),
            Refusal::PagingOfAStatedLayout => write!(
                f,
                "Arg::paged: a page is resolved digit for digit over a plain buffer's labelled \
                 dims, so a gathered mapping or a storage-tiled binding has no reading here"
            ),
            Refusal::PagingOfAnUnlabelledAxis(axis) => write!(
                f,
                "Arg::paged: {axis:?} labels no dim of this operand, so no dim holds its pages"
            ),
            Refusal::PagingOfOneDim(axis) => write!(
                f,
                "Arg::paged: {axis:?} names the pages and their positions both, on one dim"
            ),
            Refusal::PagesAlongALine(axis) => write!(
                f,
                "Arg::paged: the pages along {axis:?} lie on the innermost dim, which a line runs \
                 along, so a line would span pages"
            ),
            Refusal::PagingOfAnEmptyPool => write!(
                f,
                "Arg::paged: the pool holds no page, or its pages no position, for a table entry \
                 to read"
            ),
            Refusal::PageSplitsALine { page_size, width } => write!(
                f,
                "Arg::paged: a line of {width} positions would cross a page of {page_size}"
            ),
            Refusal::PagingMovedByRuns => write!(
                f,
                "Arg::paged: a page table is resolved a line at a time, so a packed operand or a \
                 delivery that moves runs has no reading through it"
            ),
            Refusal::IndexTableNotU32 => write!(
                f,
                "TileArgLaunch::indexed: a table holds plain u32 entries, one a line, each a \
                 coordinate"
            ),
            Refusal::IndexTableRank { keys, rank } => write!(
                f,
                "TileArgLaunch::indexed: the table has {rank} dims and {keys} keys; it holds one dim \
                 per key"
            ),
            Refusal::IndexOfAStatedLayout => write!(
                f,
                "TileArgLaunch::indexed: an entry moves the window of a plain buffer over its \
                 labelled dims, which a tensor map, a gathered, storage-tiled or paged operand is not"
            ),
            Refusal::IndexOfAnAxisNotSpanned(axis) => write!(
                f,
                "TileArgLaunch::indexed: the entry places {axis:?}, which the operand does not span"
            ),
            Refusal::IndexAlongALine(axis) => write!(
                f,
                "TileArgLaunch::indexed: the entry places {axis:?}, the innermost axis, which the \
                 operand reads more than one value a line along; bind it one value a line"
            ),
            Refusal::IndexKeyedByItsAxis(axis) => write!(
                f,
                "TileArgLaunch::indexed: the table is keyed by no axis, or by {axis:?}, the axis its \
                 entry places"
            ),
            Refusal::RunKeyedByMany(keys) => write!(
                f,
                "Indexing::Run: a run ends at the next entry along its one key, and this table has \
                 {keys} keys"
            ),
            Refusal::IndexKeysCutApart(keys) => write!(
                f,
                "Indexing: the first level cutting one of {keys:?} does not cut them all; the \
                 table is read at one step"
            ),
            Refusal::IndexedAxisCut(axis) => write!(
                f,
                "Indexing: a level cuts {axis:?}, the axis the entry places; a coordinate is the \
                 table's to state, and a run is placed whole before any level cuts it"
            ),
        }
    }
}

/// [`Refusal::BoxPastDescriptor`]'s message.
fn box_past(f: &mut Formatter<'_>, axis: Axis, edge: usize, most: usize) -> fmt::Result {
    write!(
        f,
        "TMA: {edge} along {axis:?} exceeds the {most}-per-axis box limit"
    )
}

impl std::error::Error for Refusal {}
