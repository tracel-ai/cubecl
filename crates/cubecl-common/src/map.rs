use crate::stub::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use hashbrown::HashMap;

/// A thread-safe map that allows concurrent access to values using read-write locks.
pub struct SharedStateMap<K, V> {
    state: Mutex<Option<State<K, V>>>,
}

type State<K, V> = HashMap<K, Arc<RwLock<V>>>;

/// A value in the [SharedStateMap] that provides read and write access.
pub struct SharedState<V> {
    val: Arc<RwLock<V>>,
}

impl<V> SharedState<V> {
    /// Acquires a read lock on the value, returning a read guard.
    pub fn read(&self) -> RwLockReadGuard<'_, V> {
        self.val.read().unwrap()
    }

    /// Acquires a write lock on the value, returning a write guard.
    pub fn write(&self) -> RwLockWriteGuard<'_, V> {
        self.val.write().unwrap()
    }
}

impl<K, V> Default for SharedStateMap<K, V>
where
    K: core::hash::Hash + core::cmp::PartialEq + core::cmp::Eq,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> SharedStateMap<K, V>
where
    K: core::hash::Hash + core::cmp::PartialEq + core::cmp::Eq,
{
    /// Creates a new, empty `SharedStateMap`.
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    /// Retrieves a value associated with the given key, if it exists.
    pub fn get(&self, k: &K) -> Option<SharedState<V>> {
        let mut state = self.state.lock().unwrap();
        let map = get_or_init::<K, V>(&mut state);

        match map.get(k) {
            Some(val) => Some(SharedState { val: val.clone() }),
            None => None,
        }
    }

    /// Retrieves a value associated with the given key, or inserts a new value using the provided
    /// initializer function if the key does not exist.
    pub fn get_or_init<Fn: FnMut(&K) -> V>(&self, k: &K, mut init: Fn) -> SharedState<V>
    where
        K: Clone,
    {
        let mut state = self.state.lock().unwrap();
        let map = get_or_init::<K, V>(&mut state);

        match map.get(k) {
            Some(val) => SharedState { val: val.clone() },
            None => {
                let val = init(k);
                let val = Arc::new(RwLock::new(val));
                map.insert(k.clone(), val.clone());
                SharedState { val: val.clone() }
            }
        }
    }

    /// Inserts a key-value pair into the map.
    pub fn insert(&self, k: K, v: V) {
        let mut state = self.state.lock().unwrap();
        let map = get_or_init::<K, V>(&mut state);

        map.insert(k, Arc::new(RwLock::new(v)));
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        let map = get_or_init::<K, V>(&mut state);
        map.clear();
    }
}

fn get_or_init<K, V>(state: &mut Option<State<K, V>>) -> &mut State<K, V> {
    match state {
        Some(state) => state,
        None => {
            *state = Some(State::<K, V>::default());
            state.as_mut().unwrap()
        }
    }
}
