//! How an operand's values lie in memory: its extents and strides ([`Geometry`]) and the line
//! widths they admit ([`ElemAlignment`]), the nested tiles they are laid down in
//! ([`StoragePartitioning`]) and how far apart their coarsest pieces start ([`Stagger`]), and
//! what one vector load brings ([`VectorTile`]).

pub(crate) mod elem_alignment;
pub(crate) mod geometry;
pub(crate) mod partitioning;
pub(crate) mod stagger;
pub(crate) mod vector_tile;

pub use elem_alignment::ElemAlignment;
pub use geometry::{Geometry, LineMisfit, RuntimeGeometry};
pub use partitioning::{StorageLevels, StoragePartitioning, TileMisfit};
pub use stagger::Stagger;
pub use vector_tile::VectorTile;
