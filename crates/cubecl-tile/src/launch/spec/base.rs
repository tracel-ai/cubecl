//! The comptime half of an operand.

use cubecl::zspace::SmallVec;

use crate::Delivery;
use crate::{Axis, Boundary, BoundaryPolicy, Field, Packing, Paging, Projection, Space, Storage};

/// The comptime half of an operand: its axis mapping, edge checks, packing, storage and delivery.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileSpec {
    /// How this operand's logical axes address its buffer's physical ones.
    pub projection: Projection,
    /// Out-of-bounds handling per coordinate axis; `None` or an empty list is unchecked.
    pub boundaries: SmallVec<[Option<Boundary>; Space::MAX_RANK]>,
    /// The launch's units per cube, `0` when unknown.
    pub units: usize,
    /// How this operand's values sit in its binding.
    pub(crate) packing: Packing,
    /// What this operand's storage tiles are to the windows it is read through.
    pub storage: Storage,
    /// How the buffer is stored: its stated storage tiles, finest first, as its
    /// [`StoragePartitioning`](crate::StoragePartitioning) reads back. Empty for a plain buffer.
    /// What a load of it covers is the kernel's to pick ([`VectorTile::new`](crate::VectorTile::new)).
    pub stored_tiles: Vec<(Axis, usize)>,
    /// Who moves this operand into a stage ([`delivery`](Self::delivery)).
    pub delivery: Delivery,
    /// The axes a page table resolves where the buffer is a pool of pages read as sequences
    /// ([`Arg::paged`](crate::launch::Arg::paged)); `None` for a buffer read where it lies.
    pub paging: Option<Paging>,
}

impl TileSpec {
    /// An operand's spec from its mapping, unchecked, plain, untiled, cube size unknown.
    pub fn new(projection: Projection) -> Self {
        TileSpec {
            projection,
            boundaries: SmallVec::new(),
            units: 0,
            packing: Packing::Plain,
            storage: Storage::Strided,
            stored_tiles: Vec::new(),
            delivery: Delivery::SyncPerUnit,
            paging: None,
        }
    }

    /// [`new`](Self::new) over the [`direct`](Projection::direct) mapping.
    pub fn direct(axes: &[Axis]) -> Self {
        Self::new(Projection::direct(axes))
    }

    /// The axes this operand spans, in its logical order.
    pub fn axes(&self) -> &[Axis] {
        self.projection.logical_axes()
    }

    /// This spec with one boundary mode per coordinate axis; panics on any other count.
    pub fn boundaries(mut self, boundaries: &[Option<Boundary>]) -> Self {
        let coord_rank = self.projection.coordinate_rank();
        assert!(
            boundaries.len() == coord_rank,
            "TileSpec::boundaries: {} modes stated but the operand has {coord_rank} coordinate \
             axes",
            boundaries.len()
        );
        self.boundaries = match boundaries.iter().any(Option::is_some) {
            true => SmallVec::from_slice(boundaries),
            false => SmallVec::new(),
        };
        self
    }

    /// This spec checked as `policy` says.
    pub fn boundary(self, policy: BoundaryPolicy) -> Self {
        let coord_rank = self.projection.coordinate_rank();
        match policy {
            BoundaryPolicy::Unchecked => self.boundaries(&vec![None; coord_rank]),
            // A spec has no launch to say which axes some extent could overhang: every one.
            BoundaryPolicy::Every(boundary) | BoundaryPolicy::AnyExtent(boundary) => {
                self.boundaries(&vec![Some(boundary); coord_rank])
            }
        }
    }

    /// This spec with its binding holding `u32` words of `field`-wide values.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        self.packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// This spec with its stages moved by `delivery`: each unit its own lines, landed when the fill
    /// returns or after it, or one unit the whole stage in bulk. The kernel does not change; the
    /// fill does.
    ///
    /// # Panics
    ///
    /// [`Tma`](Delivery::Tma), which moves a tensor map rather than a tensor, and
    /// [`Procedural`](Delivery::Procedural), which moves nothing.
    pub fn delivery(mut self, delivery: Delivery) -> Self {
        assert!(
            matches!(
                delivery,
                Delivery::SyncPerUnit | Delivery::AsyncPerUnit | Delivery::AsyncBulk
            ),
            "TileSpec::delivery: a tensor is moved per unit or in bulk; {delivery:?} is a tensor \
             map's or a coordinate tile's"
        );
        self.delivery = delivery;
        self
    }

    /// Whether this operand has bounds checking enabled.
    pub fn is_checked(&self) -> bool {
        self.boundaries.iter().any(Option::is_some)
    }
}
