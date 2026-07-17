/// The default hasher used by the environment's hash collections.
pub use hashbrown::DefaultHashBuilder;

/// A hash map with the environment's default hasher, usable in every
/// environment (std, no-std and wasm).
pub type HashMap<K, V, S = DefaultHashBuilder> = hashbrown::HashMap<K, V, S>;

/// A hash set with the environment's default hasher, usable in every
/// environment (std, no-std and wasm).
pub type HashSet<T, S = DefaultHashBuilder> = hashbrown::HashSet<T, S>;

pub use hashbrown::{hash_map, hash_set};
