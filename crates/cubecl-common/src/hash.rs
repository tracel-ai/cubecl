use core::hash::Hasher;
use derive_more::{Deref, DerefMut};

pub use u128 as StableHash;
use xxhash_rust::const_xxh3;

/// Stable, secure hasher.
/// # Important
/// *Do not call the [`Hasher::finish`] method. It will panic.*
/// Use [`StableHasher::finalize`] instead.
#[derive(Default, Deref, DerefMut)]
pub struct StableHasher(xxhash_rust::xxh3::Xxh3Default);

impl StableHasher {
    /// Create a new stable hasher
    pub fn new() -> Self {
        Self::default()
    }

    /// Hash one value
    pub fn hash_one<T: core::hash::Hash>(value: &T) -> StableHash {
        let mut hasher = Self::new();
        value.hash(&mut hasher);
        hasher.finalize()
    }

    /// Hash a byte slice in a const context. Slower than the non-const version, so only use when
    /// const can be guaranteed.
    pub const fn const_hash(value: &[u8]) -> StableHash {
        const_xxh3::xxh3_128(value)
    }

    /// Finalize and return the hash
    pub fn finalize(&self) -> StableHash {
        self.0.digest128()
    }
}

impl Hasher for StableHasher {
    fn finish(&self) -> u64 {
        unimplemented!("Can't finish to `u64`, use `StableHasher::finalize`")
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }
}
