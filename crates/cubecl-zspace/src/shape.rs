//! Tensor shape definition.

use super::indexing::ravel_index;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::str::FromStr;
use core::{
    ops::{Deref, DerefMut, Index, IndexMut, Range},
    slice::{Iter, IterMut, SliceIndex},
};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};

pub use crate::errors::ExpressionError;
use crate::{
    INLINE_DIMS,
    indexing::{AsIndex, AsSize},
};

/// Shape of a tensor.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct Shape {
    /// The dimensions of the tensor.
    dims: SmallVec<[usize; INLINE_DIMS]>,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Error that can occur when attempting to modify shapes.
pub enum MetadataError {
    /// The operands have different ranks.
    RankMismatch { left: usize, right: usize },
    /// A pair of dimensions are incompatible for broadcasting.
    IncompatibleDims {
        left: usize,
        right: usize,
        dim: usize,
    },
    /// Invalid dimension specified for the rank.
    OutOfBounds { dim: usize, rank: usize },
    /// A pair of shapes are incompatible for the operation.
    IncompatibleShapes { left: Shape, right: Shape },
    /// Invalid shape.
    Invalid { reason: String },
}

impl MetadataError {
    fn empty() -> Self {
        Self::Invalid {
            reason: "Shape is empty.".into(),
        }
    }
}

impl Shape {
    /// Constructs a new `Shape`.
    pub fn new<const D: usize>(dims: [usize; D]) -> Self {
        // For backward compat
        Self {
            dims: SmallVec::from_slice(&dims),
        }
    }

    /// Constructs a new `Shape` from raw backing storage. Mainly intended for macro use.
    pub fn new_raw(dims: SmallVec<[usize; INLINE_DIMS]>) -> Self {
        Self { dims }
    }

    /// Returns the total number of elements of a tensor having this shape
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the number of dimensions.
    ///
    /// Alias for `Shape::rank()`.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// Returns the rank (the number of dimensions).
    ///
    /// Alias for `Shape::num_dims()`.
    pub fn rank(&self) -> usize {
        self.num_dims()
    }

    // For compat with dims: [usize; D]
    /// Returns the dimensions of the tensor as an array.
    pub fn dims<const D: usize>(&self) -> [usize; D] {
        let mut dims = [1; D];
        dims[..D].copy_from_slice(&self.dims[..D]);
        dims
    }

    /// Change the shape to one dimensional with the same number of elements.
    pub fn flatten(mut self) -> Self {
        self.dims = SmallVec::from_slice(&[self.num_elements()]);
        self
    }

    /// Flatten the shape along a given range of dimensions.
    ///
    /// This function collapses the specified range of dimensions into a single dimension,
    /// effectively flattening the tensor in that range.
    ///
    /// # Arguments
    ///
    /// - `start_dim`: The starting dimension of the range to be flattened,
    ///   supports negative indexing.
    /// - `end_dim`: The ending dimension of the range to be flattened (inclusive),
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// A new `Shape` instance with the specified range of dimensions flattened.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cubecl_zspace::Shape;
    ///
    /// fn example() {
    ///     let shape = Shape::new([2, 3, 4]);
    ///
    ///     let flattened = shape.flatten_dims(1, 2);
    ///     println!("{flattened}");
    ///     // [2, 12]
    /// }
    /// ```
    pub fn flatten_dims(self, start_dim: impl AsIndex, end_dim: impl AsIndex) -> Self {
        let rank = self.rank();
        let start = start_dim.expect_dim_index(rank);
        let end = end_dim.expect_dim_index(rank);

        assert!(
            start <= end,
            "start_dim ({start}) must be <= than end_dim ({end})"
        );

        let existing = self.dims;

        let flattened_size = existing[start..=end].iter().product();

        let new_rank = rank - (end - start);
        let mut dims = smallvec![0; new_rank];
        dims[..start].copy_from_slice(&existing[..start]);
        dims[start] = flattened_size;
        dims[start + 1..].copy_from_slice(&existing[end + 1..]);

        Self { dims }
    }

    /// Compute the ravel index for the given coordinates.
    ///
    /// This returns the row-major order raveling:
    /// * `strides[-1] = 1`
    /// * `strides[i] = strides[i+1] * dims[i+1]`
    /// * `dim_strides = coords * strides`
    /// * `ravel = sum(dim_strides)`
    ///
    /// # Arguments
    /// - `indices`: the index for each dimension; must be the same length as `shape`.
    ///
    /// # Returns
    /// - the ravel offset index.
    pub fn ravel_index<I: AsIndex>(&self, indices: &[I]) -> usize {
        ravel_index(indices, &self.dims)
    }

    /// Convert shape dimensions to full covering ranges (0..dim) for each dimension.
    pub fn into_ranges(self) -> Vec<Range<usize>> {
        self.iter().map(|&d| 0..d).collect()
    }

    /// Construct a vector of the dims.
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.to_vec()
    }

    /// Returns an iterator over the shape dimensions.
    pub fn iter(&self) -> Iter<'_, usize> {
        self.dims.iter()
    }

    /// Mutable iterator over the dimensions.
    pub fn iter_mut(&mut self) -> IterMut<'_, usize> {
        self.dims.iter_mut()
    }

    /// Borrow the underlying dimensions slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Borrow the underlying dimensions slice mutably.
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.dims
    }

    /// Insert a dimension of `size` at position `index`.
    pub fn insert(&mut self, index: usize, size: usize) {
        self.dims.insert(index, size);
    }

    /// Remove and return the dimension at position `index` from the shape.
    pub fn remove(&mut self, index: usize) -> usize {
        self.dims.remove(index)
    }

    /// Appends a dimension of `size` to the back of the shape.
    pub fn push(&mut self, size: usize) {
        self.dims.push(size)
    }

    /// Extend the shape with the content of another shape or iterator.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = usize>) {
        self.dims.extend(iter)
    }

    /// Swap two dimensions in the shape.
    pub fn swapped(mut self, dim1: usize, dim2: usize) -> Result<Self, MetadataError> {
        if dim1 >= self.rank() {
            return Err(MetadataError::OutOfBounds {
                dim: dim1,
                rank: self.rank(),
            });
        }
        if dim2 >= self.rank() {
            return Err(MetadataError::OutOfBounds {
                dim: dim2,
                rank: self.rank(),
            });
        }
        self.dims.swap(dim1, dim2);
        Ok(self)
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), MetadataError> {
        if axes.len() != self.rank() {
            return Err(MetadataError::RankMismatch {
                left: self.rank(),
                right: axes.len(),
            });
        }
        debug_assert!(axes.iter().all(|i| i < &self.rank()));

        self.dims = axes.iter().map(|&i| self.dims[i]).collect();
        Ok(())
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permuted(mut self, axes: &[usize]) -> Result<Self, MetadataError> {
        self.permute(axes)?;
        Ok(self)
    }

    /// Repeated the specified `dim` a number of `times`.
    pub fn repeat(mut self, dim: usize, times: usize) -> Result<Shape, MetadataError> {
        if dim >= self.rank() {
            return Err(MetadataError::OutOfBounds {
                dim,
                rank: self.rank(),
            });
        }

        self.dims[dim] *= times;
        Ok(self)
    }

    /// Returns a new shape where the specified `dim` is reduced to size 1.
    pub fn reduce(mut self, dim: usize) -> Result<Shape, MetadataError> {
        if dim >= self.rank() {
            return Err(MetadataError::OutOfBounds {
                dim,
                rank: self.rank(),
            });
        }

        self.dims[dim] = 1;
        Ok(self)
    }

    /// Concatenates all shapes into a new one along the given dimension.
    pub fn cat<'a, I>(shapes: I, dim: usize) -> Result<Self, MetadataError>
    where
        I: IntoIterator<Item = &'a Shape>,
    {
        let mut iter = shapes.into_iter();

        let first = iter.next().ok_or(MetadataError::empty())?;

        if dim >= first.rank() {
            return Err(MetadataError::OutOfBounds {
                dim,
                rank: first.rank(),
            });
        }

        let mut shape = first.clone();

        for s in iter {
            if s.rank() != shape.rank() {
                return Err(MetadataError::RankMismatch {
                    left: shape.rank(),
                    right: s.rank(),
                });
            }

            if s[..dim] != shape[..dim] || s[dim + 1..] != shape[dim + 1..] {
                return Err(MetadataError::IncompatibleShapes {
                    left: shape.clone(),
                    right: s.clone(),
                });
            }

            shape[dim] += s[dim];
        }

        Ok(shape)
    }

    /// Compute the output shape for binary operations with broadcasting support.
    ///
    /// - Shapes must be of the same rank (missing dimensions are not handled automatically).
    /// - Two dimensions are compatible if they are equal, or one of them is 1.
    ///
    /// For example, a shape `[1, 1, 2, 4]` can be broadcast into `[7, 6, 2, 4]`
    /// because its axes are either equal or 1. On the other hand, a shape `[2, 2]`
    /// can *not* be broadcast into `[2, 4]`.
    pub fn broadcast(&self, other: &Self) -> Result<Self, MetadataError> {
        Self::broadcast_many([self, other])
    }

    /// Compute the broadcasted output shape across multiple input shapes.
    ///
    /// See also [broadcast](Self::broadcast).
    pub fn broadcast_many<'a, I>(shapes: I) -> Result<Self, MetadataError>
    where
        I: IntoIterator<Item = &'a Shape>,
    {
        let mut iter = shapes.into_iter();
        let mut broadcasted = iter.next().ok_or(MetadataError::empty())?.clone();
        let rank = broadcasted.rank();

        for shape in iter {
            if shape.rank() != rank {
                return Err(MetadataError::RankMismatch {
                    left: rank,
                    right: shape.rank(),
                });
            }

            for (dim, (d_lhs, &d_rhs)) in broadcasted.iter_mut().zip(shape.iter()).enumerate() {
                match (*d_lhs, d_rhs) {
                    (a, b) if a == b => {} // same
                    (1, b) => *d_lhs = b,  // broadcast to rhs
                    (_a, 1) => {}          // keep existing dimension
                    _ => {
                        return Err(MetadataError::IncompatibleDims {
                            left: *d_lhs,
                            right: d_rhs,
                            dim,
                        });
                    }
                }
            }
        }

        Ok(broadcasted)
    }

    /// Expand this shape to match the target shape, following broadcasting rules.
    pub fn expand(&self, target: Shape) -> Result<Shape, MetadataError> {
        let target_rank = target.rank();
        if self.rank() > target_rank {
            return Err(MetadataError::RankMismatch {
                left: self.rank(),
                right: target_rank,
            });
        }

        for (i, (dim_target, dim_self)) in target.iter().rev().zip(self.iter().rev()).enumerate() {
            if dim_self != dim_target && *dim_self != 1 {
                return Err(MetadataError::IncompatibleDims {
                    left: *dim_self,
                    right: *dim_target,
                    dim: target_rank - i - 1,
                });
            }
        }

        Ok(target)
    }

    /// Reshape this shape to the target shape.
    pub fn reshape<A, T>(&self, args: A) -> Result<Shape, MetadataError>
    where
        A: AsRef<[T]> + Debug,
        T: AsIndex,
    {
        let args = args.as_ref();
        let mut infer_index = None;
        let mut dims = Vec::new();

        let mut new_size = 1;

        for (idx, &s) in args.iter().enumerate() {
            let s = s.as_index();
            if s > 0 {
                let s = s as usize;
                new_size *= s;
                dims.push(s);
            } else if s == 0 {
                // We need to find the index of the 0 dimensions and
                // replace them with the actual dimension value.
                let s = self.dims[idx];
                new_size *= s;
                dims.push(s);
            } else if s == -1 {
                match infer_index {
                    None => {
                        infer_index = Some(idx);
                        // Used by / Replaced by handling later.
                        dims.push(1);
                    }
                    Some(_) => {
                        return Err(MetadataError::Invalid {
                            reason: "Repeated -1 in reshape".to_string(),
                        });
                    }
                }
            } else {
                return Err(MetadataError::Invalid {
                    reason: "The given shape cannot contain negative dimensions (other than -1)."
                        .to_string(),
                });
            }
        }

        let source_size = self.num_elements();
        match infer_index {
            None => {
                if source_size != new_size {
                    return Err(MetadataError::Invalid {
                        reason: format!(
                            "The given shape doesn't have the same number of elements as the current shape. Current shape: {self}, target shape: {dims:?}.",
                        ),
                    });
                }
            }
            Some(idx) => {
                if !source_size.is_multiple_of(new_size) {
                    return Err(MetadataError::Invalid {
                        reason: format!(
                            "Cannot infer a valid target shape. Current shape: {self}, target dimensions: {args:?}."
                        ),
                    });
                }
                dims[idx] = source_size / new_size;
            }
        }

        Ok(dims.into())
    }
}

#[macro_export]
macro_rules! shape {
    (@one $x:expr) => (1usize);
    () => (
        $crate::Shape::new_raw($crate::SmallVec::new())
    );
    ($elem:expr; $n:expr) => ({
        $crate::Shape::new_raw($crate::smallvec!($elem; $n))
    });
    ($($x:expr),+$(,)?) => ({
        $crate::Shape::new_raw($crate::smallvec!($($x),*))
    });
}

/// Compute the output shape for matrix multiplication with broadcasting support.
///
/// The last two dimensions are treated as matrices, while preceding dimensions
/// follow broadcast semantics similar to elementwise operations.
pub fn calculate_matmul_output(lhs: &Shape, rhs: &Shape) -> Result<Shape, MetadataError> {
    let rank = lhs.rank();
    if rank != rhs.rank() {
        return Err(MetadataError::RankMismatch {
            left: rank,
            right: rhs.rank(),
        });
    }

    if lhs[rank - 1] != rhs[rank - 2] {
        return Err(MetadataError::IncompatibleShapes {
            left: lhs.clone(),
            right: rhs.clone(),
        });
    }

    let mut shape = if rank > 2 {
        // Broadcast leading dims
        Shape::from(&lhs[..rank - 2]).broadcast(&Shape::from(&rhs[..rank - 2]))?
    } else {
        Shape::new([])
    };
    shape.extend([lhs[rank - 2], rhs[rank - 1]]);

    Ok(shape)
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.dims.fmt(f)
    }
}

impl FromStr for Shape {
    type Err = ExpressionError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut s = source.trim();

        const DELIMS: [(&str, &str); 2] = [("[", "]"), ("(", ")")];

        for (open, close) in DELIMS {
            if let Some(p) = s.strip_prefix(open) {
                if let Some(p) = p.strip_suffix(close) {
                    s = p.trim();
                    break;
                } else {
                    return Err(ExpressionError::ParseError {
                        message: "Unbalanced delimiters".to_string(),
                        source: source.to_string(),
                    });
                }
            }
        }

        if s.is_empty() {
            return Ok(Shape::new([]));
        }

        let dims = s
            .split(',')
            .map(|dim_str| {
                dim_str
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| ExpressionError::ParseError {
                        message: "Unable to parse shape".to_string(),
                        source: source.to_string(),
                    })
            })
            .collect::<Result<SmallVec<_>, ExpressionError>>()?;

        if dims.is_empty() {
            unreachable!("Split should have returned at least one element");
        }

        Ok(Shape { dims })
    }
}

impl<Idx> Index<Idx> for Shape
where
    Idx: SliceIndex<[usize]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.dims[index]
    }
}

impl<Idx> IndexMut<Idx> for Shape
where
    Idx: SliceIndex<[usize]>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

// Allow `&shape` to behave like a slice `&[usize]` directly
impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

// Allow `&shape` to behave like a mut slice `&mut [usize]` directly
impl DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dims
    }
}
// Allow `shape.reshape(other_shape)`.
//
// By implementing `AsRef<[usize]>`, `Shape` behaves like a slice of dimensions,
// similar to how `Vec<T>` can be passed to functions expecting a slice.
impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.dims.to_vec()
    }
}

impl<T, I> From<T> for Shape
where
    T: IntoIterator<Item = I>,
    I: AsSize,
{
    fn from(dims: T) -> Self {
        Shape {
            dims: dims.into_iter().map(|d| d.as_size()).collect(),
        }
    }
}

impl From<&Shape> for Shape {
    fn from(value: &Shape) -> Self {
        value.clone()
    }
}

impl<I: AsSize> FromIterator<I> for Shape {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Shape {
            dims: iter.into_iter().map(|it| it.as_size()).collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, reason = "useful for clarity")]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn test_shape_to_str() {
        let shape = Shape::new([2, 3, 4, 5]);
        assert_eq!(shape.to_string(), "[2, 3, 4, 5]");
    }

    #[test]
    fn test_shape_from_str() {
        assert_eq!(
            "[2, 3, 4, 5]".parse::<Shape>().unwrap(),
            Shape::new([2, 3, 4, 5])
        );
        assert_eq!(
            "(2, 3, 4, 5)".parse::<Shape>().unwrap(),
            Shape::new([2, 3, 4, 5])
        );
        assert_eq!(
            "2, 3, 4, 5".parse::<Shape>().unwrap(),
            Shape::new([2, 3, 4, 5])
        );

        assert_eq!("[2]".parse::<Shape>().unwrap(), Shape::new([2]));
        assert_eq!("(2)".parse::<Shape>().unwrap(), Shape::new([2]));
        assert_eq!("2".parse::<Shape>().unwrap(), Shape::new([2]));

        assert_eq!("[]".parse::<Shape>().unwrap(), Shape::new([]));
        assert_eq!("".parse::<Shape>().unwrap(), Shape::new([]));

        assert_eq!(
            "[".parse::<Shape>(),
            Err(ExpressionError::ParseError {
                message: "Unbalanced delimiters".to_string(),
                source: "[".to_string()
            })
        );

        assert_eq!(
            "[[1]".parse::<Shape>(),
            Err(ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "[[1]".to_string()
            })
        );
        assert_eq!(
            "[[1]]".parse::<Shape>(),
            Err(ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "[[1]]".to_string()
            })
        );
        assert_eq!(
            "[1)".parse::<Shape>(),
            Err(ExpressionError::ParseError {
                message: "Unbalanced delimiters".to_string(),
                source: "[1)".to_string()
            })
        );

        assert_eq!(
            "]".parse::<Shape>(),
            Err(ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "]".to_string()
            })
        );

        assert_eq!(
            "[a]".parse::<Shape>(),
            Err(ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "[a]".to_string()
            })
        );
    }

    #[test]
    fn num_dims_and_rank() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(4, shape.num_dims());
        assert_eq!(4, shape.rank());
    }

    #[test]
    fn num_elements() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(120, shape.num_elements());
    }

    #[test]
    #[allow(clippy::into_iter_on_ref)]
    fn test_shape_into_iter() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);

        assert_eq!(shape.into_iter().sum::<usize>(), 14);
    }

    #[test]
    fn test_into_ranges() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(shape.into_ranges(), vec![0..2, 0..3, 0..4, 0..5]);
    }

    #[test]
    fn test_to_vec() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(shape.to_vec(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_shape_index() {
        let shape = Shape::new([2, 3, 4, 5]);

        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 4);
        assert_eq!(shape[3], 5);

        // Works with ranges
        assert_eq!(shape[1..3], *&[3, 4]);
        assert_eq!(shape[1..=2], *&[3, 4]);
        assert_eq!(shape[..], *&[2, 3, 4, 5]);
    }

    #[test]
    fn test_shape_slice_methods() {
        let shape = Shape::new([2, 3, 4, 5]);

        let dim = shape.first();
        assert_eq!(dim, Some(&2));
        let dim = shape.last();
        assert_eq!(dim, Some(&5));

        assert!(!shape.is_empty());
        let shape = Shape::new([]);
        assert!(shape.is_empty());
    }

    #[test]
    fn test_shape_iter() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);

        for (d, sd) in dims.iter().zip(shape.iter()) {
            assert_eq!(d, sd);
        }
    }

    #[test]
    fn test_shape_iter_mut() {
        let mut shape = Shape::new([2, 3, 4, 5]);

        for d in shape.iter_mut() {
            *d += 1;
        }

        assert_eq!(shape.as_slice(), &[3, 4, 5, 6]);
    }

    #[test]
    fn test_shape_as_slice() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);

        assert_eq!(shape.as_slice(), dims.as_slice());

        // Deref coercion
        let shape_slice: &[usize] = &shape;
        assert_eq!(shape_slice, *&[2, 3, 4, 5]);
    }

    #[test]
    fn test_shape_as_mut_slice() {
        let mut dims = [2, 3, 4, 5];
        let mut shape = Shape::new(dims);

        let shape_mut = shape.as_mut_slice();
        assert_eq!(shape_mut, dims.as_mut_slice());
        shape_mut[1] = 6;

        assert_eq!(shape_mut, &[2, 6, 4, 5]);

        let mut shape = Shape::new(dims);
        let shape = &mut shape[..];
        shape[1] = 6;

        assert_eq!(shape, shape_mut)
    }

    #[test]
    fn test_shape_flatten() {
        let shape = Shape::new([2, 3, 4, 5]);
        assert_eq!(shape.num_elements(), 120);

        let shape = shape.flatten();
        assert_eq!(shape.num_elements(), 120);
        assert_eq!(shape.as_slice(), &[120]);
    }

    #[test]
    fn test_ravel() {
        let shape = Shape::new([2, 3, 4, 5]);

        assert_eq!(shape.ravel_index(&[0, 0, 0, 0]), 0);
        assert_eq!(
            shape.ravel_index(&[1, 2, 3, 4]),
            1 * (3 * 4 * 5) + 2 * (4 * 5) + 3 * 5 + 4
        );
    }

    #[test]
    fn test_shape_insert_remove_push() {
        let dims = [2, 3, 4, 5];
        let mut shape = Shape::new(dims);
        let size = 6;
        shape.insert(1, size);

        assert_eq!(shape, Shape::new([2, 6, 3, 4, 5]));

        let removed = shape.remove(1);
        assert_eq!(removed, size);
        assert_eq!(shape, Shape::new(dims));

        shape.push(6);
        assert_eq!(shape, Shape::new([2, 3, 4, 5, 6]));
    }

    #[test]
    fn test_shape_swap_permute() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        let shape = shape.swapped(1, 2).unwrap();

        assert_eq!(shape.as_slice(), &[2, 4, 3, 5]);

        let shape = shape.permuted(&[0, 2, 1, 3]).unwrap();
        assert_eq!(shape, Shape::new(dims));
    }

    #[test]
    #[should_panic]
    fn test_shape_swap_out_of_bounds() {
        let shape = Shape::new([2, 3, 4, 5]);

        shape.swapped(0, 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_shape_permute_incomplete() {
        let shape = Shape::new([2, 3, 4, 5]);

        shape.permuted(&[0, 2, 1]).unwrap();
    }

    #[test]
    fn test_shape_repeat() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.repeat(2, 3).unwrap();
        assert_eq!(out, Shape::new([2, 3, 12, 5]));
    }

    #[test]
    fn test_shape_repeat_invalid() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.repeat(5, 3);
        assert_eq!(out, Err(MetadataError::OutOfBounds { dim: 5, rank: 4 }));
    }

    #[test]
    fn test_shape_reduce() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.reduce(2).unwrap();
        assert_eq!(out, Shape::new([2, 3, 1, 5]));
    }

    #[test]
    fn test_shape_reduce_invalid() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.reduce(5);
        assert_eq!(out, Err(MetadataError::OutOfBounds { dim: 5, rank: 4 }));
    }

    #[test]
    fn test_shape_broadcast_binary() {
        let lhs = Shape::new([1, 1, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 1]);

        let out = lhs.broadcast(&rhs).unwrap();
        assert_eq!(out, Shape::new([7, 6, 2, 4]));
    }

    #[test]
    fn test_shape_broadcast_rank_mismatch() {
        let lhs = Shape::new([1, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 4]);

        let out = lhs.broadcast(&rhs);
        assert_eq!(out, Err(MetadataError::RankMismatch { left: 3, right: 4 }));
    }

    #[test]
    fn test_shape_broadcast_incompatible_dims() {
        let lhs = Shape::new([1, 2, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 1]);

        let out = lhs.broadcast(&rhs);
        assert_eq!(
            out,
            Err(MetadataError::IncompatibleDims {
                left: 2,
                right: 6,
                dim: 1
            })
        );
    }

    #[test]
    fn test_shape_broadcast_many() {
        let s1 = Shape::new([1, 1, 2, 4]);
        let s2 = Shape::new([7, 1, 2, 1]);
        let s3 = Shape::new([7, 6, 1, 1]);

        let out = Shape::broadcast_many([&s1, &s2, &s3]).unwrap();
        assert_eq!(out, Shape::new([7, 6, 2, 4]));
    }

    #[test]
    fn test_shape_broadcast_many_rank_mismatch() {
        let s1 = Shape::new([1, 1, 2, 4]);
        let s2 = Shape::new([7, 1, 2, 1]);
        let s3 = Shape::new([1, 6, 1]);

        let out = Shape::broadcast_many([&s1, &s2, &s3]);
        assert_eq!(out, Err(MetadataError::RankMismatch { left: 4, right: 3 }));
    }

    #[test]
    fn test_shape_broadcast_many_incompatible_dims() {
        let s1 = Shape::new([1, 1, 2, 4]);
        let s2 = Shape::new([7, 1, 2, 1]);
        let s3 = Shape::new([4, 6, 1, 1]);

        let out = Shape::broadcast_many([&s1, &s2, &s3]);
        assert_eq!(
            out,
            Err(MetadataError::IncompatibleDims {
                left: 7,
                right: 4,
                dim: 0
            })
        );
    }

    #[test]
    fn test_shape_broadcast_many_empty() {
        let out = Shape::broadcast_many(&[]);
        assert_eq!(out, Err(MetadataError::empty()));
    }

    #[test]
    fn test_shape_matmul_2d() {
        let lhs = Shape::new([2, 4]);
        let rhs = Shape::new([4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs).unwrap();
        assert_eq!(out, Shape::new([2, 2]));
    }

    #[test]
    fn test_shape_matmul_4d_broadcasted() {
        let lhs = Shape::new([1, 3, 2, 4]);
        let rhs = Shape::new([2, 1, 4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs).unwrap();
        assert_eq!(out, Shape::new([2, 3, 2, 2]));
    }

    #[test]
    fn test_shape_matmul_invalid_rank() {
        let lhs = Shape::new([3, 2, 4]);
        let rhs = Shape::new([2, 1, 4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs);
        assert_eq!(out, Err(MetadataError::RankMismatch { left: 3, right: 4 }));
    }

    #[test]
    fn test_shape_matmul_invalid_shape() {
        let lhs = Shape::new([1, 3, 2, 4]);
        let rhs = Shape::new([2, 1, 3, 2]);
        let out = calculate_matmul_output(&lhs, &rhs);
        assert_eq!(
            out,
            Err(MetadataError::IncompatibleShapes {
                left: lhs,
                right: rhs
            })
        );
    }

    #[test]
    fn test_shape_matmul_invalid_broadcast() {
        let lhs = Shape::new([1, 3, 2, 4]);
        let rhs = Shape::new([2, 2, 4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs);
        assert_eq!(
            out,
            Err(MetadataError::IncompatibleDims {
                left: 3,
                right: 2,
                dim: 1
            })
        );
    }

    #[test]
    fn test_shape_cat() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([1, 3, 4, 5]);
        let s3 = Shape::new([4, 3, 4, 5]);

        let out = Shape::cat(&[s1, s2, s3], 0).unwrap();
        assert_eq!(out, Shape::new([7, 3, 4, 5]));

        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([2, 3, 2, 5]);
        let s3 = Shape::new([2, 3, 1, 5]);

        let out = Shape::cat(&[s1, s2, s3], 2).unwrap();
        assert_eq!(out, Shape::new([2, 3, 7, 5]));
    }

    #[test]
    fn test_shape_cat_empty() {
        let out = Shape::cat(&[], 0);
        assert_eq!(out, Err(MetadataError::empty()));
    }

    #[test]
    fn test_shape_cat_dim_out_of_bounds() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([2, 3, 4, 5]);
        let out = Shape::cat(&[s1, s2], 4);
        assert_eq!(out, Err(MetadataError::OutOfBounds { dim: 4, rank: 4 }));
    }

    #[test]
    fn test_shape_cat_rank_mismatch() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([2, 3, 4, 5, 6]);
        let out = Shape::cat(&[s1, s2], 0);
        assert_eq!(out, Err(MetadataError::RankMismatch { left: 4, right: 5 }));
    }

    #[test]
    fn test_shape_cat_incompatible_shapes() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([1, 3, 4, 5]);
        let out = Shape::cat(&[s1.clone(), s2.clone()], 1);

        assert_eq!(
            out,
            Err(MetadataError::IncompatibleShapes {
                left: s1,
                right: s2
            })
        );
    }

    #[test]
    fn test_shape_expand() {
        let shape = Shape::new([1, 3, 1]);
        let expanded = Shape::new([2, 3, 4]);
        let out = shape.expand(expanded.clone()).unwrap();
        assert_eq!(out, expanded);
    }

    #[test]
    fn test_shape_expand_higher_rank() {
        let shape = Shape::new([1, 4]);
        let expanded = Shape::new([2, 3, 4]);
        let out = shape.expand(expanded.clone()).unwrap();
        assert_eq!(out, expanded);
    }

    #[test]
    fn test_shape_expand_invalid_rank() {
        let shape = Shape::new([1, 3, 1]);
        let expanded = Shape::new([3, 4]);
        let out = shape.expand(expanded);
        assert_eq!(out, Err(MetadataError::RankMismatch { left: 3, right: 2 }));
    }

    #[test]
    fn test_shape_expand_incompatible_dims() {
        let shape = Shape::new([1, 3, 2]);
        let expanded = Shape::new([2, 3, 4]);
        let out = shape.expand(expanded);
        assert_eq!(
            out,
            Err(MetadataError::IncompatibleDims {
                left: 2,
                right: 4,
                dim: 2
            })
        );
    }

    #[test]
    fn test_shape_reshape() {
        let shape = Shape::new([2, 3, 4, 5]);
        let reshaped = Shape::new([1, 2, 12, 5]);
        let out = shape.reshape(reshaped.clone()).unwrap();
        assert_eq!(out, reshaped);
    }

    #[test]
    fn test_shape_reshape_invalid() {
        let shape = Shape::new([2, 3, 4, 5]);
        let reshaped = Shape::new([2, 2, 12, 5]);
        let out = shape.reshape(reshaped.clone());
        assert_eq!(
            out,
            Err(MetadataError::Invalid {
                reason: "The given shape doesn't have the same number of elements as the current shape. Current shape: [2, 3, 4, 5], target shape: [2, 2, 12, 5].".into(),
            })
        );
    }

    #[test]
    fn test_shape_reshape_invalid_inferred() {
        let shape = Shape::new([2, 4]);
        let out = shape.reshape([-1, 3]);
        assert_eq!(
            out,
            Err(MetadataError::Invalid {
                reason: "Cannot infer a valid target shape. Current shape: [2, 4], target dimensions: [-1, 3].".into(),
            })
        );
    }

    #[test]
    fn test_flatten_dims() {
        let shape = Shape::new([2, 3, 4, 5]);
        let flattened = shape.flatten_dims(-2, 3);
        assert_eq!(flattened, Shape::new([2, 3, 20]));
    }
}
