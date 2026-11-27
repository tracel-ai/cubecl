use core::num::NonZeroU8;

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use super::ScaleDtype;
use super::code::PackedFields;

/// A level of a scheme's scales: one scale per block of values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockScale {
    /// The block of values sharing one scale.
    pub size: BlockSize,
    /// The dtype the level's scales are stored in.
    pub dtype: ScaleDtype,
}

impl BlockScale {
    /// One scale per block of `size`, stored as `dtype`.
    ///
    /// # Panics
    ///
    /// As [`BlockSize::new`], for a block given as its values.
    pub fn per_block(size: impl Into<BlockSize>, dtype: ScaleDtype) -> Self {
        Self {
            size: size.into(),
            dtype,
        }
    }

    /// One scale for the whole tensor, stored as `dtype`.
    pub fn per_tensor(dtype: ScaleDtype) -> Self {
        Self {
            size: BlockSize::TENSOR,
            dtype,
        }
    }
}

impl BlockScale {
    /// This level after `fields`; `None` where the code does not hold it.
    pub(super) fn pack(&self, fields: PackedFields) -> Option<PackedFields> {
        let listed = self.size.listed();
        let rest = [BlockExtent::VALUE, BlockExtent::WholeAxis]
            .iter()
            .position(|rest| *rest == self.size.rest)?;
        let mut fields = fields
            .with(self.dtype.to_code(), 3)?
            .with(rest as u64, 1)?
            .with(listed.len() as u64, 3)?;
        for extent in listed {
            fields = fields.with(extent.to_code(), 8)?;
        }
        Some(fields)
    }

    /// The level [`pack`](Self::pack) put next in `fields`, if it put one, and what comes after
    /// it; `None` for fields no level packs to.
    pub(super) fn unpack(fields: PackedFields) -> Option<(Option<Self>, PackedFields)> {
        let (dtype, fields) = fields.split_bits(3)?;
        if dtype == 0 {
            return Some((None, fields));
        }
        let (rest, fields) = fields.split_bits(1)?;
        let (listed, mut fields) = fields.split_bits(3)?;
        let rest = [BlockExtent::VALUE, BlockExtent::WholeAxis][rest as usize];
        let mut size = BlockSize::with_rest(&[], rest);
        for slot in size.extents.get_mut(..listed as usize)? {
            let (extent, after) = fields.split_bits(8)?;
            *slot = BlockExtent::from_code(extent);
            fields = after;
        }
        let level = Self {
            size,
            dtype: ScaleDtype::from_code(dtype)?,
        };
        Some((Some(level), fields))
    }
}

/// The block of values that shares one scale, counted on a tensor's logical axes from the last.
/// It names no extent of the tensor, so it describes one of any rank and any shape.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockSize {
    /// The block's extent along each of the last axes, the last one first.
    pub extents: [BlockExtent; MAX_DIMS],
    /// The block's extent along every axis before those.
    pub rest: BlockExtent,
}

/// How far a block reaches along one axis.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BlockExtent {
    /// This many values.
    Values(NonZeroU8),
    /// Every value of the axis, however long it is.
    WholeAxis,
}

const MAX_DIMS: usize = 5;

impl BlockExtent {
    /// A single value: the scale varies at every position of the axis.
    pub const VALUE: Self = Self::Values(NonZeroU8::MIN);

    /// How many values of an axis of `len` the block covers.
    pub fn values_of_axis(self, len: usize) -> usize {
        match self {
            Self::Values(values) => values.get() as usize,
            Self::WholeAxis => len,
        }
    }

    /// How many blocks cover an axis of `len`, the last one possibly in part.
    pub fn blocks_of_axis(self, len: usize) -> usize {
        match self {
            Self::Values(values) => len.div_ceil(values.get() as usize),
            Self::WholeAxis => 1,
        }
    }
}

impl BlockExtent {
    /// The extent as a byte, `0` for the whole axis.
    fn to_code(self) -> u64 {
        match self {
            Self::Values(values) => values.get() as u64,
            Self::WholeAxis => 0,
        }
    }

    /// The extent of a byte [`to_code`](Self::to_code) made.
    fn from_code(code: u64) -> Self {
        NonZeroU8::new(code as u8).map_or(Self::WholeAxis, Self::Values)
    }
}

impl From<u8> for BlockExtent {
    /// # Panics
    ///
    /// Where `values` is zero.
    fn from(values: u8) -> Self {
        Self::Values(NonZeroU8::new(values).expect("a block should cover a value along each axis"))
    }
}

impl BlockSize {
    /// How many of the last axes a block may reach differently along.
    pub const MAX_DIMS: usize = MAX_DIMS;

    /// The whole tensor.
    pub const TENSOR: Self = Self {
        extents: [BlockExtent::WholeAxis; MAX_DIMS],
        rest: BlockExtent::WholeAxis,
    };

    /// One row: the whole last axis, and one value along every other.
    pub const ROW: Self = Self {
        extents: [
            BlockExtent::WholeAxis,
            BlockExtent::VALUE,
            BlockExtent::VALUE,
            BlockExtent::VALUE,
            BlockExtent::VALUE,
        ],
        rest: BlockExtent::VALUE,
    };

    /// A block of this many values along each of the last axes, and of one along the others.
    ///
    /// # Panics
    ///
    /// Where a value is zero, and as [`with_rest`](Self::with_rest).
    pub fn new(values: impl AsRef<[u8]>) -> Self {
        let extents: Vec<BlockExtent> = values.as_ref().iter().copied().map(Into::into).collect();
        Self::with_rest(&extents, BlockExtent::VALUE)
    }

    /// A block reaching `extents` along the last axes, the last one last, and `rest` along every
    /// other one.
    ///
    /// # Panics
    ///
    /// Where more than [`MAX_DIMS`](Self::MAX_DIMS) extents are left after the leading ones that
    /// say `rest` again.
    pub fn with_rest(extents: &[BlockExtent], rest: BlockExtent) -> Self {
        let mut size = Self {
            extents: [rest; MAX_DIMS],
            rest,
        };
        let restated = extents.iter().take_while(|extent| **extent == rest).count();
        let listed = &extents[restated..];
        assert!(
            listed.len() <= MAX_DIMS,
            "a block reaches differently along at most {MAX_DIMS} axes, not {listed:?}"
        );
        for (slot, extent) in size.extents.iter_mut().zip(listed.iter().rev()) {
            *slot = *extent;
        }
        size
    }

    /// The block's extent along each axis of a tensor of `rank` axes.
    pub fn extents_in_rank(
        &self,
        rank: usize,
    ) -> impl DoubleEndedIterator<Item = BlockExtent> + ExactSizeIterator {
        (0..rank).map(move |axis| self.extent_along(axis, rank))
    }

    /// The block's extent along each axis of a tensor of `N` axes.
    pub fn extents_in<const N: usize>(&self) -> [BlockExtent; N] {
        core::array::from_fn(|axis| self.extent_along(axis, N))
    }

    /// How many values the block covers along each axis of `shape`.
    pub fn values(&self, shape: &[usize]) -> Vec<usize> {
        self.per_axis(shape, BlockExtent::values_of_axis)
    }

    /// How many blocks cover each axis of `shape`, which is the shape of the scale grid: one
    /// scale per block.
    pub fn num_blocks(&self, shape: &[usize]) -> Vec<usize> {
        self.per_axis(shape, BlockExtent::blocks_of_axis)
    }

    /// This block over a tensor whose axes are permuted: `axes[i]` is the axis that becomes `i`.
    ///
    /// # Panics
    ///
    /// Where `axes` is not a permutation of its indices, and as [`with_rest`](Self::with_rest).
    pub fn permuted(&self, axes: &[usize]) -> Self {
        let extents: Vec<BlockExtent> = self.extents_in_rank(axes.len()).collect();
        let permuted: Vec<BlockExtent> = axes.iter().map(|&axis| extents[axis]).collect();
        Self::with_rest(&permuted, self.rest)
    }
}

impl BlockSize {
    fn extent_from_last(&self, axis: usize) -> BlockExtent {
        self.extents.get(axis).copied().unwrap_or(self.rest)
    }

    /// The block's extent along `axis` of a tensor of `rank` axes.
    fn extent_along(&self, axis: usize, rank: usize) -> BlockExtent {
        self.extent_from_last(rank - 1 - axis)
    }

    /// `count` of the block's extent along each axis of `shape` and that axis's length.
    fn per_axis(&self, shape: &[usize], count: fn(BlockExtent, usize) -> usize) -> Vec<usize> {
        self.extents_in_rank(shape.len())
            .zip(shape)
            .map(|(extent, &len)| count(extent, len))
            .collect()
    }

    /// The extents of the last axes, the last one first, up to where they all say the rest.
    pub(super) fn listed(&self) -> &[BlockExtent] {
        let restated = self
            .extents
            .iter()
            .rev()
            .take_while(|extent| **extent == self.rest);
        &self.extents[..MAX_DIMS - restated.count()]
    }
}

impl<T: AsRef<[u8]>> From<T> for BlockSize {
    /// # Panics
    ///
    /// As [`new`](Self::new).
    fn from(value: T) -> Self {
        BlockSize::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn a_row_is_the_whole_last_axis() {
        assert_eq!(
            BlockSize::ROW,
            BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE)
        );
    }

    #[test]
    fn blocks_are_counted_from_the_last_axis() {
        assert_eq!(
            BlockSize::new([32]).extents_in(),
            [BlockExtent::VALUE, BlockExtent::from(32)]
        );
    }

    #[test]
    fn leading_unit_dimensions_say_the_rest_again() {
        assert_eq!(
            BlockSize::new([1, 1, 8, 4, 2, 3]),
            BlockSize::new([8, 4, 2, 3])
        );
    }

    #[test]
    #[should_panic(expected = "differently along at most 5 axes")]
    fn a_block_over_too_many_axes_is_refused() {
        BlockSize::new([2, 1, 1, 1, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "should cover a value")]
    fn a_block_of_no_value_is_refused() {
        BlockSize::new([0, 16]);
    }

    #[test]
    fn a_whole_axis_takes_one_scale_whatever_its_length() {
        let row = BlockSize::ROW;
        assert_eq!(row.num_blocks(&[3, 5120]), vec![3, 1]);
    }

    #[test]
    fn the_whole_tensor_takes_one_scale_whatever_its_rank() {
        assert_eq!(
            BlockSize::TENSOR.num_blocks(&[2, 2, 2, 2, 2, 8, 64]),
            vec![1; 7]
        );
    }

    #[test]
    fn a_partial_block_takes_a_scale() {
        assert_eq!(BlockSize::new([4, 32]).num_blocks(&[8, 70]), vec![2, 3]);
    }

    #[test]
    fn a_block_covers_the_whole_of_the_shapes_axis() {
        let size = BlockSize::with_rest(
            &[BlockExtent::from(4), BlockExtent::WholeAxis],
            BlockExtent::VALUE,
        );
        assert_eq!(size.values(&[2, 8, 70]), vec![1, 4, 70]);
    }

    #[test]
    fn the_extents_of_a_rank_below_the_blocks_are_its_last() {
        let extents: Vec<BlockExtent> = BlockSize::new([4, 32]).extents_in_rank(1).collect();
        assert_eq!(extents, vec![BlockExtent::from(32)]);
    }

    #[test]
    fn a_whole_axis_moves_like_any_other() {
        let row = BlockSize::ROW;
        assert_eq!(
            row.permuted(&[1, 0]),
            BlockSize::with_rest(
                &[BlockExtent::WholeAxis, BlockExtent::VALUE],
                BlockExtent::VALUE
            )
        );
    }

    #[test]
    fn permuting_keeps_the_rest() {
        let size = BlockSize::with_rest(&[BlockExtent::from(16)], BlockExtent::WholeAxis);
        assert_eq!(
            size.permuted(&[1, 0]),
            BlockSize::with_rest(
                &[BlockExtent::from(16), BlockExtent::WholeAxis],
                BlockExtent::WholeAxis
            )
        );
    }
}
