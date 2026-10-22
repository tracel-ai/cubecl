//! What a tile holds ([`TileKind`]): the one enum every verb on a tile dispatches on.

use cubecl::prelude::*;

use crate::*;

/// A tile's backing store.
/// `Clone` copies the handle, not the cells: sound only where nothing rewrites the buffer.
#[allow(clippy::large_enum_variant)]
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum TileKind {
    /// An addressable buffer with a window, in global or shared memory.
    Memory(Memory),
    /// One plane-level tile, sliced across the plane's units and never addressable.
    PlaneTile(PlaneTile),
    /// The grid of plane tiles one plane owns; only comptime coordinates select through it.
    PlanePartition(PlanePartition),
    /// A TMA tensor-map source, copied only in bulk into shared memory.
    TmaGmem(TmaData),
    /// A read-only source evaluated from logical coordinates with no backing buffer.
    Procedural(Procedural),
    /// A tile the plane holds in its units, one line to a unit ([`Lines`]).
    Lines(Lines),
}
