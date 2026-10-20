use alloc::boxed::Box;
use cubecl_environment::sync::{Mutex, Once};

/// A list that only grows, read without a lock or a write; linear, so meant for a handful of keys.
pub struct AppendOnlyMap<K, V> {
    head: Once<Box<Node<K, V>>>,
    appending: Mutex<()>,
}

struct Node<K, V> {
    key: K,
    value: V,
    next: Once<Box<Node<K, V>>>,
}

impl<K: Eq + Clone, V> AppendOnlyMap<K, V> {
    pub const fn new() -> Self {
        Self {
            head: Once::new(),
            appending: Mutex::new(()),
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.iter()
            .find(|(found, _)| *found == key)
            .map(|(_, value)| value)
    }

    /// The value of `key`, added from `value` if it is absent. `value` runs under the append lock,
    /// so it must not append to this map.
    pub fn get_or_insert_with(&self, key: &K, value: impl FnOnce() -> V) -> &V {
        if let Some(found) = self.get(key) {
            return found;
        }

        let _appending = self.appending.lock();
        // Walked again under the lock: another thread may have added the key since.
        let mut tail = &self.head;
        while let Some(node) = tail.get() {
            if node.key == *key {
                return &node.value;
            }
            tail = &node.next;
        }
        // Built outside `call_once`, where a panic would poison the slot.
        let node = Box::new(Node {
            key: key.clone(),
            value: value(),
            next: Once::new(),
        });
        &tail.call_once(|| node).value
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        core::iter::successors(self.head.get(), |node| node.next.get())
            .map(|node| (&node.key, &node.value))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    #[test]
    fn a_key_is_added_once_and_found_after() {
        let map = AppendOnlyMap::new();
        let added = AtomicUsize::new(0);
        let add = |value: u32| {
            added.fetch_add(1, Ordering::Relaxed);
            value
        };

        assert_eq!(*map.get_or_insert_with(&"a", || add(1)), 1);
        assert_eq!(*map.get_or_insert_with(&"b", || add(2)), 2);
        assert_eq!(*map.get_or_insert_with(&"a", || add(3)), 1);

        assert_eq!(added.load(Ordering::Relaxed), 2);
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), None);
        assert_eq!(
            map.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn a_value_that_panics_adds_nothing() {
        let map = AppendOnlyMap::new();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map.get_or_insert_with(&"a", || panic!("the initializer fails"));
        }));

        assert!(panicked.is_err());
        assert_eq!(map.get(&"a"), None);
        assert_eq!(*map.get_or_insert_with(&"a", || 1), 1);
    }

    #[test]
    fn racing_threads_add_each_key_once() {
        const THREADS: usize = 8;
        const KEYS: usize = 32;
        let map = AppendOnlyMap::new();
        let added = AtomicUsize::new(0);
        let barrier = Barrier::new(THREADS);

        let seen: Vec<Vec<usize>> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..THREADS)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        (0..KEYS)
                            .map(|key| {
                                *map.get_or_insert_with(&key, || {
                                    added.fetch_add(1, Ordering::Relaxed);
                                    key * 10
                                })
                            })
                            .collect()
                    })
                })
                .collect();
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });

        let expected: Vec<usize> = (0..KEYS).map(|key| key * 10).collect();
        assert!(seen.iter().all(|values| *values == expected));
        assert_eq!(added.load(Ordering::Relaxed), KEYS);
        assert_eq!(map.iter().count(), KEYS);
    }
}
