//! Where an operand's axes live: [`Projection`] maps logical axes onto buffer dims (`projection`),
//! `storage` says how the values lie in memory, `paging` reads a pool of pages as sequences, a
//! `compaction` boxes a gathered sub-tile, and a `row_arrangement` places a stage's lines.

pub(crate) mod compaction;
pub(crate) mod paging;
pub(crate) mod projection;
pub(crate) mod row_arrangement;
pub(crate) mod storage;

pub(crate) use compaction::*;
pub(crate) use paging::{PageTable, WindowCut};
pub(crate) use projection::*;
pub(crate) use row_arrangement::*;

pub use compaction::Compaction;
pub use paging::Paging;
pub use projection::{DimsBuilder, Divisor, Offset, PhysicalAxisMap, Scale, split};
pub use storage::{
    ElemAlignment, LineMisfit, RuntimeGeometry, StorageLevels, StorageMisfit, StoragePartitioning,
    TileMisfit, VectorTile,
};
