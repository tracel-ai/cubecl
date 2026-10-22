//! What a binding's derivation reads, kept with what it derived, so a launch that binds the same
//! operand again looks it up instead of deriving it again.

use std::sync::{LazyLock, Mutex};

use cubecl::prelude::CubeDim;
use cubecl::zspace::Tiling;
use hashbrown::{Equivalent, HashMap};

use super::base::{ArgData, Derived, PagedPool};
use crate::{
    Axis, BoundaryPolicy, Delivery, Geometry, Packing, Partitioning, Projection, Refusal, Space,
};

/// Every input [`Arg::realize`](super::Arg) derives an operand's spec and settled geometry from:
/// the operand as labelled, its packing as read on the device, its buffer's stored tiling, and the
/// launch's partitioning, real extents and cube. The tensor's address is left out, since the
/// derivation never reads it.
///
/// A decode step binds the same operands over the same launches every step, and deriving a tiled
/// operand's labels, partitioning and boundaries costs microseconds each time, on the thread that
/// issues every launch. Keyed by its whole input, a kept derivation cannot answer a launch it was
/// not derived for.
#[derive(PartialEq, Eq, Hash)]
pub(crate) struct Derivation {
    geometry: Geometry,
    axes: Vec<Axis>,
    batches: Vec<Axis>,
    projection: Option<Projection>,
    width: usize,
    boundary: Option<BoundaryPolicy>,
    packing: Packing,
    delivery: Delivery,
    in_stride_order: bool,
    paged: Option<PagedPool>,
    stored: Tiling,
    partitioning: Partitioning,
    space: Space,
    cube_dim: CubeDim,
}

/// A [`Derivation`] borrowed from the operand being bound: what a lookup hashes and compares, so
/// finding a kept derivation copies none of its input.
#[derive(Hash)]
pub(crate) struct DerivationView<'a> {
    geometry: &'a Geometry,
    axes: &'a [Axis],
    batches: &'a [Axis],
    projection: &'a Option<Projection>,
    width: usize,
    boundary: Option<BoundaryPolicy>,
    packing: Packing,
    delivery: Delivery,
    in_stride_order: bool,
    paged: Option<PagedPool>,
    stored: Tiling,
    partitioning: &'a Partitioning,
    space: &'a Space,
    cube_dim: CubeDim,
}

/// Derivations kept at most; past it the map starts over. A process binds a bounded set of
/// operands over a bounded set of launches, a prompt's lengths aside, which this bounds.
const KEPT: usize = 4096;

/// The derivations made so far, across every client: a derivation reads the device only through
/// the packing, which its key holds as read.
static KEPT_DERIVATIONS: LazyLock<Mutex<HashMap<Derivation, Result<Derived, Refusal>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

impl Derivation {
    /// The derivation `data` asks for, its packing read as `packing` and its buffer stored as
    /// `stored`.
    pub(crate) fn view<'a>(
        data: &'a ArgData<'a>,
        packing: Packing,
        stored: Tiling,
    ) -> DerivationView<'a> {
        DerivationView {
            geometry: &data.geometry,
            axes: data.axes,
            batches: data.batches,
            projection: &data.projection,
            width: data.width,
            boundary: data.boundary,
            packing,
            delivery: data.delivery,
            in_stride_order: data.in_stride_order,
            paged: data.paged,
            stored,
            partitioning: data.launch.partitioning(),
            space: data.launch.space(),
            cube_dim: data.launch.cube_dim(),
        }
    }
}

impl DerivationView<'_> {
    /// What this derivation derived, made by `derive` the first time it is asked for.
    pub(crate) fn derived(
        self,
        derive: impl FnOnce() -> Result<Derived, Refusal>,
    ) -> Result<Derived, Refusal> {
        if let Some(kept) = KEPT_DERIVATIONS.lock().unwrap().get(&self) {
            return kept.clone();
        }
        let derived = derive();
        let mut kept = KEPT_DERIVATIONS.lock().unwrap();
        if kept.len() >= KEPT {
            kept.clear();
        }
        kept.insert(self.owned(), derived.clone());
        derived
    }

    fn owned(self) -> Derivation {
        Derivation {
            geometry: self.geometry.clone(),
            axes: self.axes.to_vec(),
            batches: self.batches.to_vec(),
            projection: self.projection.clone(),
            width: self.width,
            boundary: self.boundary,
            packing: self.packing,
            delivery: self.delivery,
            in_stride_order: self.in_stride_order,
            paged: self.paged,
            stored: self.stored,
            partitioning: self.partitioning.clone(),
            space: self.space.clone(),
            cube_dim: self.cube_dim,
        }
    }
}

impl Equivalent<Derivation> for DerivationView<'_> {
    fn equivalent(&self, kept: &Derivation) -> bool {
        *self.geometry == kept.geometry
            && self.axes == kept.axes
            && self.batches == kept.batches
            && *self.projection == kept.projection
            && self.width == kept.width
            && self.boundary == kept.boundary
            && self.packing == kept.packing
            && self.delivery == kept.delivery
            && self.in_stride_order == kept.in_stride_order
            && self.paged == kept.paged
            && self.stored == kept.stored
            && *self.partitioning == kept.partitioning
            && *self.space == kept.space
            && self.cube_dim == kept.cube_dim
    }
}
