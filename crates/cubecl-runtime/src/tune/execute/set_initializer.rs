use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::{Any, TypeId};
use core::hash::Hash;
use cubecl_environment::collections::HashMap;

/// Initializes the tunable sets of a [`LocalTuner`](crate::tune::LocalTuner), once per initializer and
/// device: a set is built from the device it will run on — its client, its hardware properties —
/// so one device's set cannot answer for another's.
pub struct SetInitializer<ID> {
    /// Every set built, by the type of its initializer and its device (Keyed Usage), made on the
    /// first one.
    sets: Option<HashMap<(TypeId, ID), Arc<dyn Any + Send + Sync>>>,
}

impl<ID: Hash + Eq> SetInitializer<ID> {
    pub const fn new() -> Self {
        Self { sets: None }
    }

    /// The set built for `key`, if one was.
    pub fn cached<T: Any + Send + Sync>(&self, key: &(TypeId, ID)) -> Option<Arc<T>> {
        let set = self.sets.as_ref()?.get(key)?.clone();
        Some(set.downcast().expect(
            "a set built by one initializer type should always have that initializer's input and output types",
        ))
    }

    /// The set built for `key`, built by `init_set` where none was.
    pub fn initialize<T: Any + Send + Sync>(
        &mut self,
        key: (TypeId, ID),
        init_set: impl FnOnce() -> T,
    ) -> Arc<T> {
        if let Some(set) = self.cached(&key) {
            return set;
        }
        let set = Arc::new(init_set());
        self.sets
            .get_or_insert_with(HashMap::new)
            .insert(key, set.clone());
        set
    }

    /// Forget every set built, handing them back to be dropped where no lock is held.
    pub fn close(&mut self) -> Vec<Arc<dyn Any + Send + Sync>> {
        self.sets
            .iter_mut()
            .flat_map(|sets| sets.drain().map(|(_, set)| set))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: u8) -> (TypeId, u8) {
        (TypeId::of::<u32>(), id)
    }

    #[test]
    fn a_set_is_built_once_per_key() {
        let mut initializer = SetInitializer::new();
        let first = initializer.initialize(key(0), || 7u32);
        let again = initializer.initialize(key(0), || {
            unreachable!("a built set should not build again")
        });

        assert!(Arc::ptr_eq(&first, &again));
    }

    #[test]
    fn each_device_builds_its_own_set() {
        let mut initializer = SetInitializer::new();
        initializer.initialize(key(0), || 7u32);

        assert_eq!(*initializer.initialize(key(1), || 8u32), 8);
    }

    #[test]
    fn a_closed_initializer_builds_afresh() {
        let mut initializer = SetInitializer::new();
        initializer.initialize(key(0), || 7u32);
        let closed = initializer.close();

        assert_eq!(closed.len(), 1);
        assert!(initializer.cached::<u32>(&key(0)).is_none());
    }
}
