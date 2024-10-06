use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use crate::prelude::{CubeContext, CubeType, ExpandElementTyped, Init, IntoRuntime};

/// It is similar to a map, but where the keys are stored at comptime, but the values can be runtime
/// variables.
pub struct ComptimeRegistry<K, V> {
    map: Rc<RefCell<BTreeMap<K, V>>>,
}

/// To [find](ComptimeRegistry::find) an item from the [registry](ComptimeRegistry), the query must
/// be able to be translated to the actual key type.
///
/// # Example
/// If you use [u32] as key that may become [ExpandElementTyped<u32>] during the expansion, both types
/// need to implement [ComptimeRegistryQuery].
pub trait ComptimeRegistryQuery<K>: Into<K> {}

// We provide default implementations for some types.
impl ComptimeRegistryQuery<u32> for u32 {}
impl ComptimeRegistryQuery<u32> for ExpandElementTyped<u32> {}

impl Into<u32> for ExpandElementTyped<u32> {
    fn into(self) -> u32 {
        self.constant().unwrap().as_u32()
    }
}

impl<K: PartialOrd + Ord, V: CubeType + Clone> ComptimeRegistry<K, V> {
    /// Create a new registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Expand function of [Self::new].
    pub fn __expand_new(_: &mut CubeContext) -> ComptimeRegistry<K, V::ExpandType> {
        ComptimeRegistry {
            map: Rc::new(RefCell::new(BTreeMap::new())),
        }
    }

    /// Find an item in the registry.
    ///
    /// # Notes
    ///
    /// If the item isn't present the registry, the function will panic.
    pub fn find<Query: ComptimeRegistryQuery<K>>(&self, query: Query) -> V {
        let key = query.into();
        let map = self.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    /// Insert an item in the registry.
    pub fn insert<Query: ComptimeRegistryQuery<K>>(&mut self, query: Query, value: V) {
        let key = query.into();
        let mut map = self.map.as_ref().borrow_mut();

        map.insert(key, value);
    }

    /// Expand function of [Self::find].
    pub fn __expand_find<Query: ComptimeRegistryQuery<K>>(
        _context: &mut CubeContext,
        state: ComptimeRegistry<K, V::ExpandType>,
        key: Query,
    ) -> V::ExpandType {
        let key = key.into();
        let map = state.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    /// Expand function of [Self::insert].
    pub fn __expand_insert<Key: Into<K>>(
        _context: &mut CubeContext,
        state: ComptimeRegistry<K, V::ExpandType>,
        key: Key,
        value: V::ExpandType,
    ) {
        let key = key.into();
        let mut map = state.map.as_ref().borrow_mut();

        map.insert(key, value);
    }
}

impl<K: PartialOrd + Ord, V: Clone> ComptimeRegistry<K, V> {
    /// Expand method of [Self::find].
    pub fn __expand_find_method(&self, _context: &mut CubeContext, key: K) -> V {
        let map = self.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    /// Expand method of [Self::insert].
    pub fn __expand_insert_method(self, _context: &mut CubeContext, key: K, value: V) {
        let mut map = self.map.as_ref().borrow_mut();

        map.insert(key, value);
    }
}

impl<K, V> Default for ComptimeRegistry<K, V> {
    fn default() -> Self {
        Self {
            map: Rc::new(RefCell::new(BTreeMap::default())),
        }
    }
}

impl<K, V> Clone for ComptimeRegistry<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<K: PartialOrd + Ord, V: CubeType> CubeType for ComptimeRegistry<K, V> {
    type ExpandType = ComptimeRegistry<K, V::ExpandType>;
}

impl<K: PartialOrd + Ord, V> Init for ComptimeRegistry<K, V> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        self
    }
}

impl<K: PartialOrd + Ord, V: CubeType> IntoRuntime for ComptimeRegistry<K, V> {
    fn __expand_runtime_method(
        self,
        _context: &mut CubeContext,
    ) -> ComptimeRegistry<K, V::ExpandType> {
        unimplemented!("Comptime registry can't be moved to runtime.");
    }
}
