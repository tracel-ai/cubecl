//! [`Decline`]: why a launch cannot run as described, or an operand cannot be bound to it.

use core::fmt::{self, Display, Formatter};

use cubecl::zspace::{MetadataError, Tiling};

use crate::ops::matmul::leaf::memory::gather::LhsVectorMisfit;
use crate::{Delivery, Dimension, LineMisfit};

/// Why a launch cannot run as described, or an operand cannot be bound to it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Decline {
    /// [`in_stride_order`](super::Arg::in_stride_order) on a gathered or storage-tiled operand.
    StrideOrderOfAStatedLayout,
    /// A gathered mapping over a storage-tiled binding.
    GatherOfATiledBinding(Tiling),
    /// [`gathered`](super::Arg::gathered) combined with `dimensions` or `batches`.
    GatherWithLabels,
    /// The mapping addresses `mapped` axes but the operand has `rank`.
    GatherRankMismatch { mapped: usize, rank: usize },
    /// The mapping spans a dimension the launched space does not have.
    GatherOfAnUnknownDimension(Dimension),
    /// The operand's rank is smaller than its labelled block of axes.
    RankBelowLabels { rank: usize, block: usize },
    /// More leading axes than batch dimensions to label them.
    UnlabelledBatchAxes { axes: usize, dimensions: usize },
    /// An operand must address at least one coordinate.
    NoCoordinate,
    /// The buffer cannot be served `width` wide.
    WidthNotServed { width: usize, why: LineMisfit },
    /// The innermost dimension is served in vector lines but is not provably in bounds.
    UncheckableVectorEdge,
    /// A storage-tiled operand whose storage tile is the tile of `level`, and which this launch
    /// overhangs: a window inside one storage tile is read as one unmasked run, so the windows
    /// past the tensor's end would read storage tiles it does not have.
    CheckedInsideStorageTile { level: usize },
    /// A gathered mapping read in vector lines whose innermost physical axis does not step by the
    /// operand's last dimension at coefficient 1.
    GatherInnermostNotInLines,
    /// A gathered mapping that reads `dimension` off several physical axes: the operand is
    /// storage-tiled, or reads one dimension from two places.
    GatherDimensionAddressedTwice(Dimension),
    /// A grid read off the levels ([`Grid::FromLevels`](crate::launch::Grid::FromLevels)) needs a
    /// units level of one unit or the whole `plane`; this one holds `units`.
    GridNotReadOffLevels { units: u32, plane: u32 },
    /// A cube of `units` units, past the `most` the device holds.
    CubePastDevice { units: u32, most: u32 },
    /// The kernel works in planes, and the device runs no plane operations.
    PlanesUnavailable,
    /// The kernel needs one plane width, and the device's varies between `min` and `max`.
    ImprecisePlaneDim { min: u32, max: u32 },
    /// The gather leaf cannot serve an lhs vectorized along `dimension`, for the reason `why`.
    LhsVectorNotServed {
        dimension: Dimension,
        why: LhsVectorMisfit,
    },
    /// A routine does not serve the problem, for the reason `why`.
    Unserved { why: &'static str },
    /// A cube shaped `shape` along x, y and z, past the `most` the device spawns along one of them:
    /// a cube legal in units can still be too long on a dimension.
    CubeShapePastDevice {
        shape: (u32, u32, u32),
        most: (u32, u32, u32),
    },
    /// A cube with a zero dimension, which spawns no units.
    NoUnits,
    /// A grid with a zero dimension, which spawns no cubes: the runtime returns without dispatching
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
    /// A delivery whose engine, or the barrier what it moves lands on, the device does not have
    /// ([`Delivery::runs_on`](crate::launch::Delivery::runs_on)).
    DeliveryNotOnDevice(Delivery),
    /// A ring holding its next stage in the units' registers, filled by a delivery the units do
    /// not load and store themselves ([`Delivery::holds`](crate::launch::Delivery::holds)).
    PrefetchPastDelivery(Delivery),
    /// Lines `bits` long, which the copy engine does not move whole
    /// ([`Delivery::moves_lines`](crate::launch::Delivery::moves_lines)).
    LineNotCopied { bits: usize },
    /// A TMA box edge past the descriptor's per-dimension limit `most`.
    BoxPastDescriptor {
        dimension: Dimension,
        edge: usize,
        most: usize,
    },
    /// A TMA box lands its rows whole, one after the other, and the stage's blocks split the rows
    /// along `dimension`, the innermost, or stack `rows` of them where the swizzle repeats every
    /// `period`.
    BoxSplitsStageBlocks {
        dimension: Dimension,
        rows: usize,
        period: usize,
    },
    /// A TMA box lands its rows dense, at most swizzled within one span, and the stage keeps them
    /// as `why` says.
    RowsNoDescriptorLands { why: &'static str },
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages on a gathered or
    /// storage-tiled operand: a page is resolved coordinate by coordinate over a plain buffer's
    /// labelled axes.
    PagingOfAStatedLayout,
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages naming a dimension no
    /// axis of the operand is labelled with.
    PagingOfAnUnlabelledDimension(Dimension),
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages naming one axis for both
    /// the pages and their positions.
    PagingOfOneAxis(Dimension),
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages with the pages along the
    /// innermost axis, this dimension's: a line runs along it, so one line would span pages.
    PagesAlongALine(Dimension),
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages on a pool of no pages, or
    /// of pages of no positions: there is no page for a table entry to read.
    PagingOfAnEmptyPool,
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages on an operand moved by
    /// runs (a bulk copy, a tensor map): the table resolves one line at a time.
    PagingMovedByRuns,
    /// [`indexed`](crate::TileTensorLaunch::indexed) over a pool of pages whose scales span its
    /// pages or positions unplaced: they are a pool of their own, placed by the same table before
    /// they are stated.
    ScalesOfAPoolUnpaged,
    /// [`indexed`](crate::TileTensorLaunch::indexed) with a table that is not plain `u32` entries,
    /// one a line: an entry is a coordinate.
    IndexTableNotU32,
    /// [`indexed`](crate::TileTensorLaunch::indexed) with a table of `rank` axes keyed by `keys`
    /// dimensions: a table holds one axis per key.
    IndexTableRank { keys: usize, rank: usize },
    /// [`indexed`](crate::TileTensorLaunch::indexed) on a tensor map, or on a gathered,
    /// storage-tiled or paged operand: an entry moves a window of a plain buffer's labelled axes.
    IndexOfAStatedLayout,
    /// [`indexed`](crate::TileTensorLaunch::indexed) placing a dimension the operand does not span.
    IndexOfADimensionNotSpanned(Dimension),
    /// [`indexed`](crate::TileTensorLaunch::indexed) placing the innermost dimension, this one,
    /// which the operand reads more than one value a line along: an entry is an element, and a line
    /// would straddle two of them.
    IndexAlongALine(Dimension),
    /// [`indexed`](crate::TileTensorLaunch::indexed) with no key, or keyed by the dimension it
    /// places.
    IndexKeyedByItsDimension(Dimension),
    /// [`Indexing::Run`](crate::Indexing::Run) keyed by this many dimensions: a run's end is the
    /// next entry along its one key.
    RunKeyedByMany(usize),
    /// An indexed operand under a nest whose first level cutting one of these keys does not cut
    /// them all: the table is read at one step.
    IndexKeysCutApart(Vec<Dimension>),
    /// An indexed operand under a nest that cuts the dimension its entry places: a coordinate is
    /// the table's to state, and a run is placed whole before any level cuts it.
    IndexedDimensionCut(Dimension),
    /// [`codebook`](crate::TileTensorLaunch::codebook) on values not stored as indices
    /// ([`Field::Index`](crate::Field::Index)).
    CodebookOfValuesNotIndices,
    /// [`codebook`](crate::TileTensorLaunch::codebook) with a table that is not a plain one-axis
    /// buffer read one value a line.
    CodebookTableNotPlain,
    /// [`codebook`](crate::TileTensorLaunch::codebook) with a table of `entries`, fewer than the
    /// `2^bits` the indices name.
    CodebookTableShort { entries: usize, bits: usize },
    /// [`physical`](crate::layout::StoragePartitioning::physical): the tiles along `dimension` do
    /// not close its extent.
    StoredTileUnwhole {
        dimension: Dimension,
        extent: usize,
        tile: usize,
    },
    /// [`physical`](crate::layout::StoragePartitioning::physical): the storage order does not name
    /// each of the tensor's axes once.
    StorageOrderUnmatched {
        order: Vec<Dimension>,
        dimensions: Vec<Dimension>,
    },
    /// [`physical`](crate::layout::StoragePartitioning::physical): more pieces than the buffer's
    /// metadata records ([`Tiling`]).
    StoragePiecesUnrecordable(MetadataError),
}

impl Display for Decline {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Decline::StrideOrderOfAStatedLayout => write!(
                f,
                "Arg::in_stride_order: the order a buffer steps is read off strides, which a \
                 gathered mapping states for itself and a storage-tiled binding stores in tiles"
            ),
            Decline::GatherOfATiledBinding(tiling) => write!(
                f,
                "Arg::gathered: the mapping addresses the buffer's own axes, so a storage-tiled \
                 binding ({tiling:?}) has no reading here"
            ),
            Decline::GatherWithLabels => write!(
                f,
                "Arg::gathered: the mapping is stated outright, so `dimensions` and `batches` have \
                 nothing left to describe"
            ),
            Decline::GatherRankMismatch { mapped, rank } => write!(
                f,
                "Arg::gathered: the mapping addresses {mapped} axes but the operand has {rank}"
            ),
            Decline::GatherOfAnUnknownDimension(dimension) => write!(
                f,
                "Arg::gathered: the mapping spans {dimension:?}, which the launched space does not have"
            ),
            Decline::RankBelowLabels { rank, block } => write!(
                f,
                "Arg: operand rank {rank} is smaller than its labelled block of {block} axes"
            ),
            Decline::UnlabelledBatchAxes { axes, dimensions } => {
                write!(
                    f,
                    "Arg: {axes} batch axes but only {dimensions} batch dimensions given"
                )
            }
            Decline::NoCoordinate => {
                write!(f, "Arg: an operand must address at least one coordinate")
            }
            Decline::WidthNotServed { width, why } => {
                write!(
                    f,
                    "Arg::vectorize: this operand cannot be served {width} wide: {why}"
                )
            }
            Decline::UncheckableVectorEdge => write!(
                f,
                "Arg: the innermost dimension is served in vector lines but is not provably in bounds \
                 (it overhangs its tiling at an extent the width does not divide, under a clamp or \
                 any extent, or it spans several dimensions, or its map reaches below its start or past \
                 any extent stated here); serve it at a width dividing its extent or scalar, or \
                 state BoundaryPolicy::Unchecked if the launch proves its vector lines are in \
                 bounds"
            ),
            Decline::CheckedInsideStorageTile { level } => write!(
                f,
                "Arg: this storage-tiled operand's storage tile is the tile of level {level}, \
                 where a window is read as one unmasked run, and the launch overhangs it, so a \
                 window past the tensor's end would read storage tiles it does not have; tile \
                 the launch so it divides the operand, or tile it coarser than the storage tile"
            ),
            Decline::GatherInnermostNotInLines => write!(
                f,
                "Arg::gathered: the innermost physical axis must step by the operand's last \
                 dimension at coefficient 1 (it is addressed in vector lines)"
            ),
            Decline::GatherDimensionAddressedTwice(dimension) => write!(
                f,
                "Arg::gathered: dimension {dimension:?} addresses several physical axes, so it is \
                 either storage-tiled (a gathered operand must be untiled gmem) or read off two \
                 places at once"
            ),
            Decline::GridNotReadOffLevels { units, plane } => write!(
                f,
                "Launcher: a grid read off the levels needs a units level of one unit or the \
                 whole plane ({plane}), got {units}"
            ),
            Decline::CubePastDevice { units, most } => write!(
                f,
                "Launcher: a cube of {units} units, but the device holds at most {most}"
            ),
            Decline::PlanesUnavailable => {
                write!(
                    f,
                    "the kernel works in planes, and the device runs no plane operations"
                )
            }
            Decline::ImprecisePlaneDim { min, max } => write!(
                f,
                "the kernel needs one plane width, and the device's varies from {min} to {max}"
            ),
            Decline::LhsVectorNotServed { dimension, why } => {
                write!(
                    f,
                    "contract gather: lhs vectorized along {dimension:?}: {why}"
                )
            }
            Decline::Unserved { why } => f.write_str(why),
            Decline::CubeShapePastDevice { shape, most } => write!(
                f,
                "Launcher: a cube shaped {shape:?}, but the device spawns at most {most:?}"
            ),
            Decline::NoUnits => write!(
                f,
                "Launcher: the cube has a zero dimension and spawns no units"
            ),
            Decline::NoCubes => write!(
                f,
                "Launcher: the grid has a zero dimension and spawns no cubes"
            ),
            Decline::GridPastDevice { count, most } => write!(
                f,
                "Launcher: a grid of {count:?} cubes, but the device launches at most {most:?}"
            ),
            Decline::SharedMemoryPastDevice { bytes, most } => write!(
                f,
                "Launcher: a cube holding {bytes} bytes of shared memory, but the device gives \
                 one at most {most}"
            ),
            Decline::FillersWithoutBarriers { fillers } => write!(
                f,
                "Launcher: {fillers} plane(s) are set aside to fill a walk's stages, and this \
                 device carries no barrier type for the two roles to meet on"
            ),
            Decline::DeliveryNotOnDevice(delivery) => write!(
                f,
                "Delivery: this device has no engine for {delivery:?}, or no barrier for what it \
                 moves to land on"
            ),
            Decline::PrefetchPastDelivery(delivery) => write!(
                f,
                "Delivery: a stage held in the units' registers is one they load and store \
                 themselves, and {delivery:?} moves it without them"
            ),
            Decline::LineNotCopied { bits } => write!(
                f,
                "Delivery: the copy engine moves lines of 4, 8 or 16 bytes, and these are {bits} \
                 bits"
            ),
            Decline::BoxPastDescriptor {
                dimension,
                edge,
                most,
            } => write!(
                f,
                "TMA: {edge} along {dimension:?} exceeds the {most}-per-dimension box limit"
            ),
            Decline::BoxSplitsStageBlocks {
                dimension,
                rows,
                period,
            } => write!(
                f,
                "TMA: a box lands its rows whole and in order, and the stage's blocks split \
                 {dimension:?} or stack {rows} rows apiece where the swizzle repeats every {period}"
            ),
            Decline::RowsNoDescriptorLands { why } => write!(
                f,
                "TMA: a box lands its rows dense, at most swizzled within a span, and the stage \
                 holds {why}"
            ),
            Decline::PagingOfAStatedLayout => write!(
                f,
                "TileTensorLaunch::indexed: a page is resolved coordinate by coordinate over a \
                 plain buffer's labelled axes, so a gathered mapping or a storage-tiled binding \
                 has no reading here"
            ),
            Decline::PagingOfAnUnlabelledDimension(dimension) => write!(
                f,
                "TileTensorLaunch::indexed: {dimension:?} labels no axis of this operand, so no axis \
                 holds its pages"
            ),
            Decline::PagingOfOneAxis(dimension) => write!(
                f,
                "TileTensorLaunch::indexed: {dimension:?} names the pages and their positions both, on \
                 one axis"
            ),
            Decline::PagesAlongALine(dimension) => write!(
                f,
                "TileTensorLaunch::indexed: the pages along {dimension:?} lie on the innermost axis, \
                 which a line runs along, so a line would span pages"
            ),
            Decline::PagingOfAnEmptyPool => write!(
                f,
                "TileTensorLaunch::indexed: the pool holds no page, or its pages no position, for \
                 a table entry to read"
            ),
            Decline::ScalesOfAPoolUnpaged => write!(
                f,
                "TileTensorLaunch::indexed: the scales of a pool of pages span its pages or \
                 positions, so they are a pool of their own, placed by the same table before they \
                 are stated"
            ),
            Decline::PagingMovedByRuns => write!(
                f,
                "TileTensorLaunch::indexed: a page table is resolved a line at a time, so a \
                 delivery that moves runs has no reading through it"
            ),
            Decline::CodebookOfValuesNotIndices => write!(
                f,
                "TileTensorLaunch::codebook: a codebook is read through values stored as indices \
                 (`Field::Index`)"
            ),
            Decline::CodebookTableNotPlain => write!(
                f,
                "TileTensorLaunch::codebook: the table is a plain one-axis buffer, one value a line"
            ),
            Decline::CodebookTableShort { entries, bits } => write!(
                f,
                "TileTensorLaunch::codebook: {bits}-bit indices name {} entries; the table holds \
                 {entries}",
                1usize << bits
            ),
            Decline::IndexTableNotU32 => write!(
                f,
                "TileTensorLaunch::indexed: a table holds plain u32 entries, one a line, each a \
                 coordinate"
            ),
            Decline::IndexTableRank { keys, rank } => write!(
                f,
                "TileTensorLaunch::indexed: the table has {rank} axes and {keys} keys; it holds \
                 one axis per key"
            ),
            Decline::IndexOfAStatedLayout => write!(
                f,
                "TileTensorLaunch::indexed: an entry moves the window of a plain buffer over its \
                 labelled axes, which a tensor map, a gathered, storage-tiled or paged operand is not"
            ),
            Decline::IndexOfADimensionNotSpanned(dimension) => write!(
                f,
                "TileTensorLaunch::indexed: the entry places {dimension:?}, which the operand does not \
                 span"
            ),
            Decline::IndexAlongALine(dimension) => write!(
                f,
                "TileTensorLaunch::indexed: the entry places {dimension:?}, the innermost dimension, which \
                 the operand reads more than one value a line along; bind it one value a line"
            ),
            Decline::IndexKeyedByItsDimension(dimension) => write!(
                f,
                "TileTensorLaunch::indexed: the table is keyed by no dimension, or by {dimension:?}, the \
                 dimension its entry places"
            ),
            Decline::RunKeyedByMany(keys) => write!(
                f,
                "Indexing::Run: a run ends at the next entry along its one key, and this table has \
                 {keys} keys"
            ),
            Decline::IndexKeysCutApart(keys) => write!(
                f,
                "Indexing: the first level cutting one of {keys:?} does not cut them all; the \
                 table is read at one step"
            ),
            Decline::IndexedDimensionCut(dimension) => write!(
                f,
                "Indexing: a level cuts {dimension:?}, the dimension the entry places; a coordinate is the \
                 table's to state, and a run is placed whole before any level cuts it"
            ),
            Decline::StoredTileUnwhole {
                dimension,
                extent,
                tile,
            } => write!(
                f,
                "StoragePartitioning::physical: dimension {dimension:?} runs {extent}, which is not a whole \
                 number of {tile}-wide tiles"
            ),
            Decline::StorageOrderUnmatched { order, dimensions } => write!(
                f,
                "StoragePartitioning::physical: the tiles follow {order:?}, but the tensor's axes \
                 are {dimensions:?}: the order names each once"
            ),
            Decline::StoragePiecesUnrecordable(why) => write!(
                f,
                "StoragePartitioning::physical: the buffer's metadata cannot record its pieces: \
                 {why:?}"
            ),
        }
    }
}

impl std::error::Error for Decline {}
