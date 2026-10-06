use alloc::{vec, vec::Vec};
use core::hash::Hash;
use cubecl_environment::collections::HashMap;

/// Scoped map for optimizations: a key resolves in the innermost scope that holds it.
#[derive(Debug)]
pub struct ScopedMap<K, V> {
    scopes: Vec<HashMap<K, V>>,
}

impl<K, V> ScopedMap<K, V> {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::default()],
        }
    }
}

impl<K, V> Default for ScopedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq, V> ScopedMap<K, V> {
    fn depth(&self) -> usize {
        self.scopes.len() - 1
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let depth = self.depth();
        self.scopes[depth].insert(key, value)
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::default());
    }

    pub fn pop_scope(&mut self) {
        assert!(self.depth() > 0, "Tried popping root scope");
        self.scopes.pop();
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.scopes.iter().rev().find_map(|scope| scope.get(key))
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.scopes
            .iter_mut()
            .rev()
            .find_map(|scope| scope.get_mut(key))
    }
}
