use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
    INLINE_DIMS, MetadataError,
    shape::Shape,
    strides::Strides,
    tiling::{MAX_FRAGMENTS, Tiling},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct Metadata {
    pub shape: Shape,
    pub strides: Strides,
    /// How many fragments each logical dim is stored as; untiled by default.
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
        tiling.logical_rank(self.rank())?;
        self.tiling = tiling;
        Ok(self)
    }

    /// Whether any logical dim is stored as more than one fragment.
    pub fn is_tiled(&self) -> bool {
        self.tiling.is_tiled()
    }

    /// How many dims this buffer stands for: its rank, less the extra fragments
    /// the tiling splits dims into.
    ///
    /// # Errors
    ///
    /// When the tiling does not fit this rank: see [`Tiling::logical_rank`].
    pub fn logical_rank(&self) -> Result<usize, MetadataError> {
        self.tiling.logical_rank(self.rank())
    }

    /// The extents this buffer stands for: each logical dim's fragments
    /// multiplied back together, in logical order. An untiled metadata gives its
    /// shape unchanged.
    ///
    /// # Errors
    ///
    /// When the tiling does not fit this rank: see [`Tiling::logical_rank`].
    pub fn logical_shape(&self) -> Result<Shape, MetadataError> {
        let fragments = self.tiling.fragments(self.logical_rank()?);
        let mut extents = SmallVec::<[usize; INLINE_DIMS]>::from_elem(1, fragments.len());
        // A dim's fragments are spread through the buffer rather than adjacent,
        // so walk the levels in the order the storage was laid out, coarsest
        // first, and each dim collects its own.
        let dims = self.shape.as_slice();
        let mut physical = 0;
        for level in 0..MAX_FRAGMENTS {
            for (dim, &count) in fragments.iter().enumerate() {
                if level < count {
                    extents[dim] *= dims[physical];
                    physical += 1;
                }
            }
        }
        Ok(Shape::new_raw(extents))
    }

    /// The dim-changing ops do not carry a tiling yet: they refuse rather than
    /// return counts over dims that moved.
    fn assert_untiled(&self, op: &str) {
        assert!(
            !self.is_tiled(),
            "Metadata::{op} on a storage-tiled tensor is not supported: {:?}",
            self.tiling
        );
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The shape, to rewrite in place. A storage-tiled tensor's dims are its tiling's fragments,
    /// which a rewrite would leave stale, so it refuses like the dim-changing ops do.
    pub fn shape_mut(&mut self) -> &mut Shape {
        self.assert_untiled("shape_mut");
        &mut self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// The strides, to rewrite in place. Refuses on a storage-tiled tensor, as
    /// [`shape_mut`](Self::shape_mut) does: its strides step its tiles, and a rewrite would
    /// keep the tiling over strides that no longer do.
    pub fn strides_mut(&mut self) -> &mut Strides {
        self.assert_untiled("strides_mut");
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
        self.assert_untiled("swap");
        debug_assert!(dim0 < self.rank(), "dim0 is out of bounds");
        debug_assert!(dim1 < self.rank(), "dim1 is out of bounds");
        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), MetadataError> {
        self.assert_untiled("permute");
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
        self.assert_untiled("insert");
        self.shape.insert(index, shape);
        self.strides.insert(index, stride);
    }

    /// Remove and return the dimension at position `index` from the metadata.
    pub fn remove(&mut self, index: usize) -> (usize, usize) {
        self.assert_untiled("remove");
        let shape = self.shape.remove(index);
        let stride = self.strides.remove(index);
        (shape, stride)
    }

    /// Appends a dimension of `shape` with `stride` to the back of the metadata.
    pub fn push(&mut self, shape: usize, stride: usize) {
        self.assert_untiled("push");
        self.shape.push(shape);
        self.strides.push(stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `[b, m, k]` operand stored `[Bs, Mx, Ky, Mi, Kj]`: the two tiled dims
    /// collect a fragment from each level, the untiled one drops out after the
    /// first.
    #[test]
    fn logical_shape_multiplies_a_dim_s_fragments_back_together() {
        let meta = Metadata::new(
            [2, 128, 344, 32, 32],
            [128 * 344 * 1024, 344 * 1024, 1024, 32, 1],
        )
        .with_tiling(Tiling::new(&[1, 2, 2]).unwrap())
        .unwrap();

        assert_eq!(meta.logical_rank(), Ok(3));
        assert_eq!(meta.logical_shape(), Ok(Shape::new([2, 4096, 11008])));
    }

    /// The untiled case is the identity, so a caller reasoning in logical dims
    /// need not ask whether the tensor is tiled first.
    #[test]
    fn an_untiled_metadata_stands_for_its_own_shape() {
        let meta = Metadata::new([2, 4096, 11008], [4096 * 11008, 11008, 1]);

        assert_eq!(meta.logical_rank(), Ok(3));
        assert_eq!(meta.logical_shape(), Ok(meta.shape.clone()));
    }

    /// Three levels deep on one dim, one on another: the fragment counts need
    /// not match, and the physical order stays level-major.
    #[test]
    fn dims_tiled_to_different_depths_each_collect_their_own() {
        let meta = Metadata::new([4, 8, 2, 4, 2], [1; 5])
            .with_tiling(Tiling::new(&[3, 2]).unwrap())
            .unwrap();

        assert_eq!(meta.logical_shape(), Ok(Shape::new([4 * 2 * 2, 8 * 4])));
    }

    /// A buffer too short to hold the fragments is the caller pairing the wrong
    /// tensor with the wrong description, and says so rather than guessing.
    #[test]
    fn a_buffer_too_short_for_the_tiling_is_refused() {
        let mut meta = Metadata::new([2, 128, 344, 32, 32], [1; 5])
            .with_tiling(Tiling::new(&[1, 2, 2]).unwrap())
            .unwrap();
        meta.shape.remove(4);
        meta.strides.remove(4);
        meta.shape.remove(3);
        meta.strides.remove(3);

        assert!(meta.logical_rank().is_err());
        assert!(meta.logical_shape().is_err());
    }
}
