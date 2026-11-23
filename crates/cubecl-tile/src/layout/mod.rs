//! Where an operand's dimensions live: [`Projection`] maps dimensions onto buffer axes
//! (`projection`), `storage` says how the values lie in memory, `paging` reads a pool of pages as
//! sequences, a `compaction` boxes a gathered sub-tile, and a `row_arrangement` places a stage's
//! lines.

pub(crate) mod compaction;
pub(crate) mod paging;
pub(crate) mod projection;
pub(crate) mod reads;
pub(crate) mod row_arrangement;
pub(crate) mod storage;

pub(crate) use compaction::*;
pub(crate) use paging::{PageTable, WindowCut};
pub(crate) use projection::*;
pub(crate) use row_arrangement::*;

pub use compaction::Compaction;
pub use paging::{PageEntry, Paging};
pub use projection::{AxesBuilder, Divisor, Offset, PhysicalAxisMap, Scale, split};
pub use storage::{
    ElemAlignment, LineMisfit, Run, RuntimeGeometry, Stagger, StorageLevels, StoragePartitioning,
    TileMisfit, VectorTile,
};
