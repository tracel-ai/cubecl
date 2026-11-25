use crate::server::LazyRead;
use alloc::sync::Arc;
use core::ops::Range;

/// A range of a [`LazyRead`]'s bytes, held for bytes or a view of them to read until dropped.
pub struct LazyReadHold {
    read: Arc<LazyRead>,
    range: Range<usize>,
}

impl LazyReadHold {
    /// The read it holds bytes of.
    pub fn read(&self) -> &Arc<LazyRead> {
        &self.read
    }

    /// The bytes of the read it holds.
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// A hold of `range` within these bytes, for a view of them.
    ///
    /// # Assumptions
    ///
    /// - `range` lies within this hold's bytes.
    pub fn narrowed(&self, range: Range<usize>) -> LazyReadHold {
        self.read
            .hold(self.range.start + range.start..self.range.start + range.end)
    }
}

impl LazyReadHold {
    pub(crate) fn new(read: Arc<LazyRead>, range: Range<usize>) -> Self {
        Self { read, range }
    }
}

impl Drop for LazyReadHold {
    fn drop(&mut self) {
        self.read.let_go(&self.range);
    }
}

impl core::fmt::Debug for LazyReadHold {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LazyReadHold")
            .field("range", &self.range)
            .finish()
    }
}
