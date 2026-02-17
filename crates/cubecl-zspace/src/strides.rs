use core::ops::{Deref, DerefMut};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::{INLINE_DIMS, MetadataError, indexing::AsSize};

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

    /// Insert a dimension of `stride` at position `index`.
    pub fn insert(&mut self, index: usize, stride: usize) {
        self.dims.insert(index, stride);
    }

    /// Remove and return the dimension at position `index` from the strides.
    pub fn remove(&mut self, index: usize) -> usize {
        self.dims.remove(index)
    }

    /// Appends a dimension of `stride` to the back of the strides.
    pub fn push(&mut self, stride: usize) {
        self.dims.push(stride)
    }

    /// Extend the strides with the content of another shape or iterator.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = usize>) {
        self.dims.extend(iter)
    }

    /// Reorder the strides dimensions according to the permutation of `axes`.
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

    /// Reorder the strides dimensions according to the permutation of `axes`.
    pub fn permuted(mut self, axes: &[usize]) -> Result<Self, MetadataError> {
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
