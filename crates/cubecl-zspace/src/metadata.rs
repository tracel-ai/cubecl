use serde::{Deserialize, Serialize};

use crate::{MetadataError, shape::Shape, strides::Strides, tiling::Tiling};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct Metadata {
    pub shape: Shape,
    pub strides: Strides,
    /// Which logical dim each physical dim is a fragment of; untiled by default.
    /// The shape and strides stay physical. See [`Tiling`].
    pub tiling: Tiling,
}

impl Metadata {
    pub fn new(shape: impl Into<Shape>, strides: impl Into<Strides>) -> Self {
        let shape = shape.into();
        let strides = strides.into();
        debug_assert_eq!(
            shape.rank(),
            strides.rank(),
            "Rank of shape and strides must be the same"
        );

        Self {
            shape,
            strides,
            tiling: Tiling::UNTILED,
        }
    }

    /// This metadata with `tiling` labelling its physical dims.
    ///
    /// # Errors
    ///
    /// When `tiling` does not describe this rank: see [`Tiling::new`].
    pub fn with_tiling(mut self, tiling: Tiling) -> Result<Self, MetadataError> {
        if tiling.is_tiled() {
            Tiling::new(&tiling.labels(self.rank()))?;
        }
        self.tiling = tiling;
        Ok(self)
    }

    /// Whether any physical dim is a fragment of a logical one.
    pub fn is_tiled(&self) -> bool {
        self.tiling.is_tiled()
    }

    /// The dim-changing ops do not carry a tiling yet: they refuse rather than
    /// return labels over dims that moved.
    fn untiled_for(&self, op: &str) {
        assert!(
            !self.is_tiled(),
            "Metadata::{op} on a storage-tiled tensor is not supported: {:?}",
            self.tiling
        );
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn shape_mut(&mut self) -> &mut Shape {
        &mut self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn strides_mut(&mut self) -> &mut Strides {
        &mut self.strides
    }

    pub fn rank(&self) -> usize {
        self.num_dims()
    }

    pub fn num_dims(&self) -> usize {
        self.shape.num_dims()
    }

    /// Returns the total number of elements of a tensor having this shape
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    pub fn swapped(mut self, dim0: usize, dim1: usize) -> Self {
        self.swap(dim0, dim1);
        self
    }

    pub fn swap(&mut self, dim0: usize, dim1: usize) {
        self.untiled_for("swap");
        debug_assert!(dim0 < self.rank(), "dim0 is out of bounds");
        debug_assert!(dim1 < self.rank(), "dim1 is out of bounds");
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), MetadataError> {
        self.untiled_for("permute");
        self.shape.permute(axes)?;
        self.strides.permute(axes)?;

        Ok(())
    }

    pub fn permuted(mut self, axes: &[usize]) -> Result<Self, MetadataError> {
        self.permute(axes)?;
        Ok(self)
    }

    /// Insert a dimension of `shape` with `stride` at position `index`.
    pub fn insert(&mut self, index: usize, shape: usize, stride: usize) {
        self.untiled_for("insert");
        self.shape.insert(index, shape);
        self.strides.insert(index, stride);
    }

    /// Remove and return the dimension at position `index` from the metadata.
    pub fn remove(&mut self, index: usize) -> (usize, usize) {
        self.untiled_for("remove");
        let shape = self.shape.remove(index);
        let stride = self.strides.remove(index);
        (shape, stride)
    }

    /// Appends a dimension of `shape` with `stride` to the back of the metadata.
    pub fn push(&mut self, shape: usize, stride: usize) {
        self.untiled_for("push");
        self.shape.push(shape);
        self.strides.push(stride);
    }
}
