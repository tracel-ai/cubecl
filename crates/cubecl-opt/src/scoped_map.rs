use alloc::{vec, vec::Vec};

/// Scoped map for optimizations
/// uses `Vec` as internal storage to allow non-hash keys. Lookup will actually be faster generally,
/// since `HashMap` only starts ammortizing for large amounts of values, and for small sets linear
/// lookup tends to be faster.
#[derive(Debug)]
pub struct ScopedMap<K, V> {
    scopes: Vec<Vec<(K, V)>>,
}

impl<K, V> ScopedMap<K, V> {
    pub fn new() -> Self {
        Self {
            scopes: vec![vec![]],
        }
    }
}

impl<K, V> Default for ScopedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: PartialEq, V> ScopedMap<K, V> {
    fn depth(&self) -> usize {
        self.scopes.len() - 1
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let depth = self.depth();
        let scope = &mut self.scopes[depth];
        if let Some((_, existing)) = scope.iter_mut().find(|(k, _)| k == &key) {
            Some(core::mem::replace::<V>(existing, value))
        } else {
            scope.push((key, value));
            None
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(Default::default());
    }

    pub fn pop_scope(&mut self) {
        assert!(self.depth() > 0, "Tried popping root scope");
        self.scopes.pop();
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        for scope in self.scopes.iter().rev() {
            if let Some((_, val)) = scope.iter().find(|(k, _)| k == key) {
                return Some(val);
            }
        }
        None
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some((_, val)) = scope.iter_mut().find(|(k, _)| k == key) {
                return Some(val);
            }
        }
        None
    }
}
