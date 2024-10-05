use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

use crate::prelude::{CubeContext, CubeType, ExpandElementTyped, Init, IntoRuntime};

pub struct ConstMap<K, V> {
    map: Rc<RefCell<BTreeMap<K, V>>>,
}

pub trait ConstMapQuery<K>: Into<K> {}

impl ConstMapQuery<u32> for u32 {}
impl ConstMapQuery<u32> for ExpandElementTyped<u32> {}

impl Into<u32> for ExpandElementTyped<u32> {
    fn into(self) -> u32 {
        self.constant().unwrap().as_u32()
    }
}

impl<K: PartialOrd + Ord, V: CubeType + Clone> ConstMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn __expand_new(_: &mut CubeContext) -> ConstMap<K, V::ExpandType> {
        ConstMap {
            map: Rc::new(RefCell::new(BTreeMap::new())),
        }
    }

    pub fn get<Query: ConstMapQuery<K>>(&self, query: Query) -> V {
        let key = query.into();
        let map = self.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    pub fn insert<Query: ConstMapQuery<K>>(&mut self, query: Query, value: V) {
        let key = query.into();
        let mut map = self.map.as_ref().borrow_mut();

        map.insert(key, value);
    }

    pub fn __expand_get<Query: ConstMapQuery<K>>(
        _context: &mut CubeContext,
        state: ConstMap<K, V::ExpandType>,
        key: Query,
    ) -> V::ExpandType {
        let key = key.into();
        let map = state.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    pub fn __expand_insert<Key: Into<K>>(
        _context: &mut CubeContext,
        state: ConstMap<K, V::ExpandType>,
        key: Key,
        value: V::ExpandType,
    ) {
        let key = key.into();
        let mut map = state.map.as_ref().borrow_mut();

        map.insert(key, value);
    }
}

impl<K: PartialOrd + Ord, V: Clone> ConstMap<K, V> {
    pub fn __expand_get_method(&self, _context: &mut CubeContext, key: K) -> V {
        let map = self.map.as_ref().borrow();

        map.get(&key).unwrap().clone()
    }

    pub fn __expand_insert_method(self, _context: &mut CubeContext, key: K, value: V) {
        let mut map = self.map.as_ref().borrow_mut();

        map.insert(key, value);
    }
}

impl<K, V> Default for ConstMap<K, V> {
    fn default() -> Self {
        Self {
            map: Rc::new(RefCell::new(BTreeMap::default())),
        }
    }
}

impl<K, V> Clone for ConstMap<K, V> {
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<K: PartialOrd + Ord, V: CubeType> CubeType for ConstMap<K, V> {
    type ExpandType = ConstMap<K, V::ExpandType>;
}

impl<K: PartialOrd + Ord, V> Init for ConstMap<K, V> {
    fn init(self, _context: &mut crate::prelude::CubeContext) -> Self {
        self
    }
}

impl<K: PartialOrd + Ord, V: CubeType> IntoRuntime for ConstMap<K, V> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> ConstMap<K, V::ExpandType> {
        unimplemented!("Const map doesn't exist at compile time");
    }
}
