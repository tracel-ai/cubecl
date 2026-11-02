use alloc::format;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{
    INLINE_DIMS, MetadataError,
    shape::Shape,
    strides::Strides,
    tiling::{MAX_LOGICAL_DIMS, Tiling},
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
        if !self.is_tiled() {
            return Ok(self.shape.clone());
        }
        let extents = self
            .tiling
            .physical_dims(self.logical_rank()?)
            .iter()
            .map(|physical| physical.iter().map(|&p| self.shape[p]).product())
            .collect();
        Ok(Shape::new_raw(extents))
    }

    /// This metadata, untiled, restated in `reference`'s tiling: each logical dim split into the
    /// fragments `reference` stores it as and laid out in `reference`'s physical order, so the two
    /// walk their buffers' physical dims together. A split is one stride per fragment, so the
    /// buffer it describes is the same.
    ///
    /// # Errors
    ///
    /// When this metadata is tiled itself, or its shape is not `reference`'s logical shape.
    pub fn tiled_like(&self, reference: &Metadata) -> Result<Metadata, MetadataError> {
        let logical = reference.logical_shape()?;
        if self.is_tiled() {
            return Err(MetadataError::Invalid {
                reason: format!("{:?} is tiled already", self.tiling),
            });
        }
        if self.shape != logical {
            return Err(MetadataError::IncompatibleShapes {
                left: self.shape.clone(),
                right: logical,
            });
        }
        let mut strides = SmallVec::from_elem(0, reference.rank());
        let physical_dims = reference.tiling.physical_dims(logical.rank());
        for (physical, &stride) in physical_dims.iter().zip(self.strides.iter()) {
            // The finest fragment steps as the logical dim does; each coarser one steps over
            // every finer fragment of its dim.
            let mut step = stride;
            for &p in physical.iter().rev() {
                strides[p] = step;
                step *= reference.shape[p];
            }
        }
        Ok(Metadata {
            shape: reference.shape.clone(),
            strides: Strides::new_raw(strides),
            tiling: reference.tiling,
        })
    }

    /// Rewriting the shape or the strides in place would leave a tiling over dims that moved: the
    /// writers refuse rather than describe the wrong buffer.
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

    /// Swap logical dims `dim0` and `dim1`. On a storage-tiled tensor each moves with all its
    /// fragments, as [`permute`](Self::permute) moves them.
    pub fn swap(&mut self, dim0: usize, dim1: usize) {
        if !self.is_tiled() {
            debug_assert!(dim0 < self.rank(), "dim0 is out of bounds");
            debug_assert!(dim1 < self.rank(), "dim1 is out of bounds");
            self.shape.swap(dim0, dim1);
            self.strides.swap(dim0, dim1);
            return;
        }
        let mut axes: SmallVec<[usize; MAX_LOGICAL_DIMS]> =
            (0..self.known_logical_rank()).collect();
        axes.swap(dim0, dim1);
        self.permute(&axes)
            .expect("two logical dims swapped are a permutation of them");
    }

    /// Reorder the logical dims according to the permutation of `axes`. On a storage-tiled
    /// tensor each logical dim moves with all its fragments, and the physical dims stay
    /// level-major in the new logical order: the tiling is reordered with them, and no value
    /// moves.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), MetadataError> {
        if !self.is_tiled() {
            self.shape.permute(axes)?;
            self.strides.permute(axes)?;
            return Ok(());
        }
        let rank = self.logical_rank()?;
        if axes.len() != rank {
            return Err(MetadataError::RankMismatch {
                left: rank,
                right: axes.len(),
            });
        }
        let fragments = self.tiling.fragments(rank);
        let permuted: SmallVec<[usize; MAX_LOGICAL_DIMS]> =
            axes.iter().map(|&axis| fragments[axis]).collect();
        let tiling = Tiling::new(&permuted)?;
        let from = self.tiling.physical_dims(rank);
        let to = tiling.physical_dims(rank);
        let mut physical = SmallVec::<[usize; INLINE_DIMS]>::from_elem(0, self.rank());
        for (dim, &axis) in axes.iter().enumerate() {
            for (&new, &old) in to[dim].iter().zip(&from[axis]) {
                physical[new] = old;
            }
        }
        self.shape.permute(&physical)?;
        self.strides.permute(&physical)?;
        self.tiling = tiling;
        Ok(())
    }

    pub fn permuted(mut self, axes: &[usize]) -> Result<Self, MetadataError> {
        self.permute(axes)?;
        Ok(self)
    }

    /// Insert a logical dimension of `shape` with `stride` at logical position `index`. On a
    /// storage-tiled tensor it is one fragment, placed among the coarsest.
    pub fn insert(&mut self, index: usize, shape: usize, stride: usize) {
        let physical = match self.is_tiled() {
            false => index,
            true => {
                let mut fragments = self.tiling.fragments(self.known_logical_rank());
                fragments.insert(index, 1);
                self.tiling = Tiling::new(&fragments)
                    .expect("a tiled tensor has room for one more untiled dim");
                self.tiling.physical_dims(fragments.len())[index][0]
            }
        };
        self.shape.insert(physical, shape);
        self.strides.insert(physical, stride);
    }

    /// Remove and return the logical dimension at position `index` from the metadata.
    ///
    /// # Panics
    ///
    /// On a dim stored in more than one fragment: it has no one extent and stride to return.
    pub fn remove(&mut self, index: usize) -> (usize, usize) {
        let physical = match self.is_tiled() {
            false => index,
            true => {
                let rank = self.known_logical_rank();
                let mut fragments = self.tiling.fragments(rank);
                assert_eq!(
                    fragments.remove(index),
                    1,
                    "Metadata::remove of a storage-tiled dim: {:?}",
                    self.tiling
                );
                let physical = self.tiling.physical_dims(rank)[index][0];
                self.tiling = Tiling::new(&fragments).expect("fewer dims fit a tiling");
                physical
            }
        };
        let shape = self.shape.remove(physical);
        let stride = self.strides.remove(physical);
        (shape, stride)
    }

    /// Appends a logical dimension of `shape` with `stride` to the back of the metadata.
    pub fn push(&mut self, shape: usize, stride: usize) {
        self.insert(self.known_logical_rank(), shape, stride);
    }

    /// The logical rank of a metadata built through [`with_tiling`](Self::with_tiling), which
    /// checked it.
    fn known_logical_rank(&self) -> usize {
        self.logical_rank()
            .expect("a metadata's tiling describes its own rank")
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

    /// A `[k, n] = [8, 12]` matrix stored in `(4, 3)` tiles, the tiles and each tile's rows
    /// row-major: `[2, 4, 4, 3]` over `[k/4, n/3, 4, 3]`.
    fn tiled_matrix() -> Metadata {
        Metadata::new([2, 4, 4, 3], [48, 12, 3, 1])
            .with_tiling(Tiling::new(&[2, 2]).unwrap())
            .unwrap()
    }

    /// Where logical `coords` lie in the buffer `meta` describes: each coordinate split into its
    /// dim's fragments, finest last.
    fn offset(meta: &Metadata, coords: &[usize]) -> usize {
        let physical_dims = meta.tiling.physical_dims(coords.len());
        let mut offset = 0;
        for (physical, &coord) in physical_dims.iter().zip(coords) {
            let mut rest = coord;
            for &p in physical.iter().rev() {
                offset += (rest % meta.shape[p]) * meta.strides[p];
                rest /= meta.shape[p];
            }
        }
        offset
    }

    /// A transposed tiled matrix is the same buffer: every value is reached at the swapped
    /// coordinates, so a transpose needs no copy.
    #[test]
    fn a_swap_moves_each_dim_with_its_fragments() {
        let meta = tiled_matrix();
        let swapped = meta.clone().swapped(0, 1);

        assert_eq!(swapped.logical_shape(), Ok(Shape::new([12, 8])));
        assert_eq!(swapped.tiling.fragments(2).as_slice(), &[2, 2]);
        for i in 0..8 {
            for j in 0..12 {
                assert_eq!(offset(&swapped, &[j, i]), offset(&meta, &[i, j]));
            }
        }
    }

    /// Dims tiled to different depths keep the physical order level-major after a permute.
    #[test]
    fn a_permute_keeps_every_value_in_place() {
        let meta = Metadata::new([2, 2, 3, 4, 2, 3], [144, 72, 24, 6, 3, 1])
            .with_tiling(Tiling::new(&[1, 2, 3]).unwrap())
            .unwrap();
        let axes = [2, 0, 1];
        let permuted = meta.clone().permuted(&axes).unwrap();

        assert_eq!(permuted.logical_shape(), Ok(Shape::new([18, 2, 8])));
        for b in 0..2 {
            for k in 0..8 {
                for n in 0..18 {
                    assert_eq!(offset(&permuted, &[n, b, k]), offset(&meta, &[b, k, n]));
                }
            }
        }
    }

    /// An untiled dim inserted and removed again leaves the tiled ones as they were.
    #[test]
    fn an_untiled_dim_goes_in_and_out_of_a_tiled_metadata() {
        let meta = tiled_matrix();
        let mut batched = meta.clone();
        batched.insert(0, 5, 0);

        assert_eq!(batched.logical_shape(), Ok(Shape::new([5, 8, 12])));
        for i in 0..8 {
            for j in 0..12 {
                assert_eq!(offset(&batched, &[3, i, j]), offset(&meta, &[i, j]));
            }
        }
        assert_eq!(batched.remove(0), (5, 0));
        assert_eq!(batched, meta);
    }

    /// A tiled dim has no one extent and stride.
    #[test]
    #[should_panic(expected = "storage-tiled dim")]
    fn a_tiled_dim_is_not_removed() {
        tiled_matrix().remove(1);
    }

    /// A matrix in rows restated in a tiled matrix's layout reaches each value where its rows
    /// put it, so the two walk the same physical dims over the same values.
    #[test]
    fn rows_restated_in_tiles_reach_the_same_values() {
        let rows = Metadata::new([8, 12], [12, 1]);
        let restated = rows.tiled_like(&tiled_matrix()).unwrap();

        assert_eq!(restated.shape, tiled_matrix().shape);
        for i in 0..8 {
            for j in 0..12 {
                assert_eq!(offset(&restated, &[i, j]), i * 12 + j);
            }
        }
    }

    /// A broadcast dim splits into broadcast fragments.
    #[test]
    fn a_broadcast_row_restated_in_tiles_stays_broadcast() {
        let bias = Metadata::new([8, 12], [0, 1]);
        let restated = bias.tiled_like(&tiled_matrix()).unwrap();

        for i in 0..8 {
            for j in 0..12 {
                assert_eq!(offset(&restated, &[i, j]), j);
            }
        }
    }

    #[test]
    fn restating_refuses_another_shape_or_a_tiled_metadata() {
        assert!(
            Metadata::new([8, 6], [6, 1])
                .tiled_like(&tiled_matrix())
                .is_err()
        );
        assert!(tiled_matrix().tiled_like(&tiled_matrix()).is_err());
    }
}
