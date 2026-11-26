use super::PersistentSliceReserver;
use crate::memory_management::memory_pool::padded_size;

/// What a [`PersistentSliceReserver`] holds, as its readers see it: a handle
/// on the pool that reads each part only when asked.
#[derive(Clone, Copy)]
pub struct PersistentView<'a> {
    pool: &'a PersistentSliceReserver,
}

impl<'a> PersistentView<'a> {
    /// A view of `pool`.
    pub fn new(pool: &'a PersistentSliceReserver) -> Self {
        Self { pool }
    }

    /// Whether the pool holds a slice of the padded size `size_bytes` takes,
    /// free or not.
    pub fn has_size_bytes(&self, size_bytes: u64) -> bool {
        padded_size(size_bytes, self.pool.alignment)
            .is_some_and(|effective_size| self.pool.sizes.contains_key(&effective_size))
    }
}
