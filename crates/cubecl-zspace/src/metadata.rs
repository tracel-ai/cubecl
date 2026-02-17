use serde::{Deserialize, Serialize};

use crate::{MetadataError, shape::Shape, strides::Strides};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Metadata {
    pub shape: Shape,
    pub strides: Strides,
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

        Self { shape, strides }
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
        debug_assert!(dim0 < self.rank(), "dim0 is out of bounds");
        debug_assert!(dim1 < self.rank(), "dim1 is out of bounds");
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), MetadataError> {
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
        self.shape.insert(index, shape);
        self.strides.insert(index, stride);
    }

    /// Remove and return the dimension at position `index` from the metadata.
    pub fn remove(&mut self, index: usize) -> (usize, usize) {
        let shape = self.shape.remove(index);
        let stride = self.strides.remove(index);
        (shape, stride)
    }

    /// Appends a dimension of `shape` with `stride` to the back of the metadata.
    pub fn push(&mut self, shape: usize, stride: usize) {
        self.shape.push(shape);
        self.strides.push(stride);
    }
}
