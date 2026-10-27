//! The axis-agnostic tile DSL engine: a [`Space`] cut into [`Level`]s forms a [`Partitioning`],
//! whose loops hand out [`Region`]s; the [`Launcher`] reads the grid off the same levels.

// `#[cube]` expands to `cubecl::` paths, which the language crate serves.
extern crate cubecl_core as cubecl;

pub(crate) mod algebra;
mod comptime_only;
pub mod dialect;
pub mod launch;
pub mod layout;
pub mod ops;
pub mod space;
pub mod stage;
pub(crate) mod tile;

/// What a [`Tile`] can be in the kernel.
pub mod kind {
    pub use crate::tile::accumulator::planes_output::PlanesOutput;
    pub use crate::tile::accumulator::smem_accumulation::SmemAccumulation;
    pub use crate::tile::accumulator::smem_cyclic::SmemCyclicAccumulation;
    pub use crate::tile::accumulator::smem_slots::SmemSlotsAccumulation;
    pub use crate::tile::kind::memory::base::{Boundary, Schedule, Storage, Write};
    pub use crate::tile::kind::memory::global::GlobalOperand;
    pub use crate::tile::kind::memory::view::decoded::Decoded;
    pub use crate::tile::kind::memory::view::masked::{Masked, MaskedMut};
    pub use crate::tile::packing::base::Field;
    /// A field's codes in and out of words, exposed for the round-trip tests.
    #[doc(hidden)]
    pub use crate::tile::packing::view::{pack_line, unpack_line};
}

/// Tiles whose values are computed from their coordinates.
pub mod procedural {
    pub use crate::tile::kind::procedural::affine::{AffineCoordinate, affine_along};
    pub use crate::tile::kind::procedural::base::{Reads, Recipe, RecipeCoords, RecipeExpand};
    pub use crate::tile::kind::procedural::constant::Constant;
    pub use crate::tile::kind::procedural::kind::Procedural;
    pub use crate::tile::kind::procedural::normalization::{DivGuard, TapSupport};
    pub use crate::tile::kind::procedural::phase::Phase;
    pub use crate::tile::kind::procedural::product::{Product, product_of};
    pub use crate::tile::kind::procedural::separable::Factors;
    pub use crate::tile::kind::procedural::sum::{Sum, sum_of};
}

pub(crate) use comptime_only::comptime_only;

// Flat crate namespace for `use crate::*`.
#[allow(unused_imports)]
pub(crate) use {algebra::*, launch::*, layout::*, ops::*, space::*, stage::*, tile::*};

pub use algebra::monoid::{Carrier, Monoid};
pub use algebra::semiring::Semiring;
pub use launch::arg::tensor::{TileArg, TileArgLaunch};
pub use launch::base::Launcher;
pub use launch::spec::TileSpec;
pub use layout::projection::Projection;
pub use layout::storage::Geometry;
pub use ops::matmul::instruction::{Instruction, MmaOperands};
pub use ops::matmul::register_block::RegisterBlock;
pub use space::axis::Axis;
pub use space::base::Space;
pub use space::partition::base::Partitioning;
pub use space::partition::level::Level;
pub use space::partition::levels::Levels;
pub use space::region::{Leaves, Planes, Region};
pub use space::walk::base::Walk;
pub use stage::base::StageStorage;
pub use stage::pipeline::base::Stages;
pub use tile::accumulator::base::Scratch;
pub use tile::base::{Tile, TileExpand};
pub use tile::factor::{Factor, FactorExpand};
pub use tile::index::Indexing;
pub use tile::slices::{AxisSlices, AxisSlicesExpand};
