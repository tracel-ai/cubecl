use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use cubecl_ir::Scope;

use crate::prelude::{CubeDebug, CubeType, ExpandElementTyped, Init};

/// It is similar to a map, but where the keys are stored at comptime, but the values can be runtime
/// variables.
pub struct Registry<K, V> {
    map: Rc<RefCell<BTreeMap<K, V>>>,
}

/// To [find](Registry::find) an item from the [registry](Registry), the query must
/// be able to be translated to the actual key type.
///
/// # Example
///
/// If you use [u32] as key that may become [`ExpandElementTyped<u32>`] during the expansion, both types
/// need to implement [RegistryQuery].
pub trait RegistryQuery<K>: Into<K> {}

// We provide default implementations for some types.
impl RegistryQuery<u32> for u32 {}
impl RegistryQuery<u32> for ExpandElementTyped<u32> {}

impl From<ExpandElementTyped<u32>> for u32 {
    fn from(val: ExpandElementTyped<u32>) -> Self {
        val.constant().unwrap().as_u32()
    }
}

impl<K: PartialOrd + Ord + core::fmt::Debug, V: CubeType + Clone> Registry<K, V> {
    /// Create a new registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Expand function of [Self::new].
    pub fn __expand_new(_: &mut Scope) -> Registry<K, V::ExpandType> {
        Registry {
            map: Rc::new(RefCell::new(BTreeMap::new())),
        }
    }

    /// Find an item in the registry.
    ///
    /// # Notes
    ///
    /// If the item isn't present in the registry, the function will panic.
    pub fn find<Query: RegistryQuery<K>>(&self, query: Query) -> V {
        let key = query.into();
        let map = self.map.as_ref().borrow();

        match map.get(&key) {
            Some(val) => val.clone(),
            None => panic!("No value found for key {key:?}"),
        }
    }

    /// Insert an item in the registry.
    pub fn insert<Query: RegistryQuery<K>>(&mut self, query: Query, value: V) {
        let key = query.into();
        let mut map = self.map.as_ref().borrow_mut();

        map.insert(key, value);
    }

    /// Expand function of [Self::find].
    pub fn __expand_find<Query: RegistryQuery<K>>(
        _scope: &mut Scope,
        state: Registry<K, V::ExpandType>,
        key: Query,
    ) -> V::ExpandType {
        let key = key.into();
        let map = state.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    /// Expand function of [Self::insert].
    pub fn __expand_insert<Key: Into<K>>(
        _scope: &mut Scope,
        state: Registry<K, V::ExpandType>,
        key: Key,
        value: V::ExpandType,
    ) {
        let key = key.into();
        let mut map = state.map.as_ref().borrow_mut();

        map.insert(key, value);
    }
}

impl<K: PartialOrd + Ord + core::fmt::Debug, V: Clone> Registry<K, V> {
    /// Expand method of [Self::find].
    pub fn __expand_find_method(&self, _scope: &mut Scope, key: K) -> V {
        let map = self.map.as_ref().borrow();

        match map.get(&key) {
            Some(val) => val.clone(),
            None => panic!("No value found for key {key:?}"),
        }
    }

    /// Expand method of [Self::insert].
    pub fn __expand_insert_method(self, _scope: &mut Scope, key: K, value: V) {
        let mut map = self.map.as_ref().borrow_mut();

        map.insert(key, value);
    }
}

impl<K, V> Default for Registry<K, V> {
    fn default() -> Self {
        Self {
            map: Rc::new(RefCell::new(BTreeMap::default())),
        }
    }
}

impl<K, V> Clone for Registry<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<K: PartialOrd + Ord, V: CubeType> CubeType for Registry<K, V> {
    type ExpandType = Registry<K, V::ExpandType>;
}

impl<K: PartialOrd + Ord, V> Init for Registry<K, V> {
    fn init(self, _scope: &mut crate::ir::Scope) -> Self {
        self
    }
}

impl<K: PartialOrd + Ord, V> CubeDebug for Registry<K, V> {}
