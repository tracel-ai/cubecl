//! The logical coordinate space a tile lives in, and how a level of it splits (`partition`).

pub(crate) mod axis;
pub(crate) mod base;
pub(crate) mod coords;
pub(crate) mod extent;
pub(crate) mod matrix;
pub(crate) mod partition;
pub(crate) mod path;
pub(crate) mod region;
pub(crate) mod share;
pub(crate) mod step;
pub(crate) mod walk;

pub(crate) use axis::*;
pub(crate) use base::*;
pub(crate) use extent::*;
pub(crate) use matrix::*;
pub(crate) use partition::*;
pub(crate) use path::Path;
pub(crate) use region::*;
pub(crate) use share::*;
pub(crate) use step::*;
pub(crate) use walk::*;

pub use coords::Coords;
pub use coords::CoordsExpand;
pub use extent::Shape;
pub use partition::cube_order::{CubeOrder, swizzle};
pub use partition::level::{ComputeScope, Count, Coverage, Spread};
pub use partition::plane_order::{PlaneOrder, UnitCube};
pub use partition::table::LevelTable;
pub use share::UnitShare;
pub use walk::portion::Portion;
