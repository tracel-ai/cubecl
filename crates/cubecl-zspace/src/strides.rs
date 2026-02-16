use core::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{INLINE_DIMS, ShapeError, indexing::AsSize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub struct Strides {
    dims: SmallVec<[usize; INLINE_DIMS]>,
}

impl Strides {
    pub fn new(dims: &[usize]) -> Self {
        // For backward compat
        Self {
            dims: SmallVec::from_slice(dims),
        }
    }

    pub fn new_raw(dims: SmallVec<[usize; INLINE_DIMS]>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
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

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(&mut self, axes: &[usize]) -> Result<(), ShapeError> {
        if axes.len() != self.rank() {
            return Err(ShapeError::RankMismatch {
                left: self.rank(),
                right: axes.len(),
            });
        }
        debug_assert!(axes.iter().all(|i| i < &self.rank()));

        self.dims = axes.iter().map(|&i| self.dims[i]).collect();
        Ok(())
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permuted(mut self, axes: &[usize]) -> Result<Self, ShapeError> {
        self.permute(axes)?;
        Ok(self)
    }
}

impl Deref for Strides {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl DerefMut for Strides {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dims
    }
}

#[macro_export]
macro_rules! strides {
    (@one $x:expr) => (1usize);
    () => (
        $crate::Strides::new_raw($crate::SmallVec::new())
    );
    ($elem:expr; $n:expr) => ({
        $crate::Strides::new_raw($crate::smallvec!($elem; $n))
    });
    ($($x:expr),+$(,)?) => ({
        $crate::Strides::new_raw($crate::smallvec!($($x),*))
    });
}

impl<T, I> From<T> for Strides
where
    T: IntoIterator<Item = I>,
    I: AsSize,
{
    fn from(dims: T) -> Self {
        Strides {
            dims: dims.into_iter().map(|d| d.as_size()).collect(),
        }
    }
}

impl From<&Strides> for Strides {
    fn from(value: &Strides) -> Self {
        value.clone()
    }
}

impl<I: AsSize> FromIterator<I> for Strides {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Strides {
            dims: iter.into_iter().map(|it| it.as_size()).collect(),
        }
    }
}
