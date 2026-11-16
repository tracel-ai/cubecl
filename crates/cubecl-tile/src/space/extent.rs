//! One dimension's size ([`Extent`]) and every dimension's at once ([`Shape`]).

use crate::{Dimension, DimensionMap};

/// One dimension's size: `Static` is comptime, `Dynamic` is resolved in-kernel from the tensor shape.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Extent {
    Static(usize),
    Dynamic,
}

impl Extent {
    /// The comptime size; panics on `Dynamic`.
    pub(crate) fn get(self) -> usize {
        match self {
            Extent::Static(n) => n,
            Extent::Dynamic => {
                panic!("Extent::get: this dimension is Dynamic; its size is only known at runtime")
            }
        }
    }

    pub(crate) fn is_dynamic(self) -> bool {
        matches!(self, Extent::Dynamic)
    }

    /// The extent two spaces sharing a dimension broadcast to when they are
    /// [`merge`](crate::Space::merge)d.
    pub(crate) fn broadcast(self, other: Extent) -> Extent {
        match (self, other) {
            (Extent::Static(1), b) => b,
            (a, Extent::Static(1)) => a,
            (Extent::Dynamic, _) | (_, Extent::Dynamic) => Extent::Dynamic,
            (Extent::Static(a), Extent::Static(b)) if a == b => Extent::Static(a),
            _ => panic!("Space::merge: dimension appears with conflicting extents"),
        }
    }
}

/// Every dimension with its comptime extent, in canonical order.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shape {
    extents: DimensionMap<Extent>,
}

impl Shape {
    pub(crate) fn new(extents: &[(Dimension, Extent)]) -> Self {
        Shape {
            extents: DimensionMap::new(extents),
        }
    }

    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    pub fn dimension_at(&self, i: usize) -> Dimension {
        self.extents.dimension_at(i)
    }

    pub fn position(&self, dimension: Dimension) -> usize {
        self.extents.position(dimension)
    }

    pub fn contains(&self, dimension: Dimension) -> bool {
        self.extents.contains(dimension)
    }

    pub fn dimensions(&self) -> impl Iterator<Item = Dimension> + '_ {
        self.extents.dimensions()
    }

    pub(crate) fn extent_raw(&self, dimension: Dimension) -> Extent {
        self.extents.get(dimension)
    }

    /// The dimension's comptime size; panics on a dynamic dimension.
    pub fn extent(&self, dimension: Dimension) -> usize {
        self.extent_raw(dimension).get()
    }

    pub fn extent_at(&self, i: usize) -> usize {
        self.extent(self.dimension_at(i))
    }

    pub fn is_dynamic(&self, dimension: Dimension) -> bool {
        self.extent_raw(dimension).is_dynamic()
    }

    /// Every dimension is [`Static`](Extent::Static), so a walk over it is fully comptime.
    pub(crate) fn is_static(&self) -> bool {
        self.dimensions()
            .all(|dimension| !self.is_dynamic(dimension))
    }

    /// The listed dimensions with their extents, in the order listed.
    pub(crate) fn subspace(&self, dimensions: &[Dimension]) -> Shape {
        let entries: Vec<_> = dimensions
            .iter()
            .map(|&a| (a, self.extent_raw(a)))
            .collect();
        Shape::new(&entries)
    }
}
