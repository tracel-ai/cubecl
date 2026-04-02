use core::hash::{BuildHasher, Hasher};
use hashbrown::HashSet;

use crate::kernel::KernelIdFast;

/// Tracks which kernels have already been execution-checked.
///
/// Uses a [`HashSet`] with a custom [`IdentityHasherBuilder`] to skip redundant hashing —
/// [`KernelIdFast`] already carries a precomputed `u128` hash, so the identity hasher
/// passes it through directly rather than hashing it again.
///
/// Unlike storing raw `u128` hashes, this stores full [`KernelIdFast`] values so that
/// `hashbrown` can fall back to [`KernelId`] equality on the (astronomically unlikely)
/// chance of a 128-bit hash collision.
pub struct KernelExecutionChecks {
    checked: HashSet<KernelIdFast, IdentityHasherBuilder>,
}

impl KernelExecutionChecks {
    /// Creates a new, empty set of execution checks.
    pub const fn new() -> Self {
        let checked = HashSet::with_hasher(IdentityHasherBuilder);
        Self { checked }
    }

    /// Returns `true` if the given kernel has already been checked.
    pub fn should_check(&mut self, id: &KernelIdFast) -> bool {
        if self.checked.contains(id) {
            return false;
        }

        self.checked.insert(id.clone());
        true
    }
}

/// A [`BuildHasher`] that produces [`IdentityHasher`] instances.
///
/// Intended exclusively for use with types that carry a precomputed hash (like
/// [`KernelIdFast`]). Do not use with types whose [`Hash`] impl writes arbitrary data —
/// the hasher assumes exactly one 16-byte write.
struct IdentityHasherBuilder;

impl BuildHasher for IdentityHasherBuilder {
    type Hasher = IdentityHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher {
            state: 0,
            written: false,
        }
    }
}

/// A no-op hasher that expects a single `u128` write and passes it through.
///
/// On [`finish`](Hasher::finish), the 128-bit state is folded into 64 bits via XOR
/// of the upper and lower halves. This is sufficient because the input is already
/// a well-distributed hash.
///
/// # Panics (debug builds only)
///
/// - [`write`](Hasher::write) panics if called more than once or with a slice that
///   is not exactly 16 bytes.
/// - [`finish`](Hasher::finish) panics if called before any write.
#[derive(Default)]
struct IdentityHasher {
    state: u128,
    written: bool,
}

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        debug_assert!(
            self.written,
            "IdentityHasher::finish called without a write"
        );
        let lo = self.state as u64;
        let hi = (self.state >> 64) as u64;
        lo ^ hi
    }

    fn write(&mut self, bytes: &[u8]) {
        debug_assert!(
            !self.written,
            "IdentityHasher expects exactly one write of a u128"
        );
        debug_assert!(
            bytes.len() == 16,
            "IdentityHasher expects a u128 (16 bytes)"
        );
        self.state = u128::from_ne_bytes(bytes.try_into().unwrap());
        self.written = true;
    }
}

#[cfg(feature = "std")]
mod with_std {
    use core::cell::RefCell;
    use std::thread_local;

    use super::*;

    thread_local! {
        static STATE: RefCell<KernelExecutionChecks> = RefCell::new(KernelExecutionChecks::new());
    }

    pub fn should_check(id: &KernelIdFast) -> bool {
        STATE.with_borrow_mut(|state| state.should_check(id))
    }
}
#[cfg(feature = "std")]
pub use with_std::should_check;

#[cfg(not(feature = "std"))]
mod without_std {
    use super::*;

    static STATE: spin::Mutex<KernelExecutionChecks> =
        spin::Mutex::new(KernelExecutionChecks::new());

    pub fn perform_check(id: &KernelIdFast) -> bool {
        let mut state = STATE.lock();

        state.should_check(id)
    }
}
#[cfg(not(feature = "std"))]
pub use without_std::perform_check;

#[cfg(test)]
mod tests {
    use super::*;
    use core::hash::Hash;

    /// Helper to run a value through the identity hasher and return the u64 result.
    fn identity_hash(value: u128) -> u64 {
        let mut hasher = IdentityHasherBuilder.build_hasher();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn finish_xors_upper_and_lower_halves() {
        let lo: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let hi: u64 = 0x1234_5678_9ABC_DEF0;
        let value = (hi as u128) << 64 | lo as u128;

        assert_eq!(identity_hash(value), lo ^ hi);
    }

    #[test]
    fn zero_hash_works() {
        // Ensure 0 is not treated as a sentinel.
        assert_eq!(identity_hash(0), 0);
    }

    #[test]
    fn different_values_produce_different_hashes() {
        assert_ne!(identity_hash(1), identity_hash(2));
    }

    #[test]
    fn deterministic() {
        let value: u128 = 0x0123_4567_89AB_CDEF_FEDC_BA98_7654_3210;
        assert_eq!(identity_hash(value), identity_hash(value));
    }

    #[test]
    #[should_panic(expected = "exactly one write")]
    fn panics_on_double_write() {
        let mut hasher = IdentityHasherBuilder.build_hasher();
        let bytes = [0u8; 16];
        hasher.write(&bytes);
        hasher.write(&bytes);
    }

    #[test]
    #[should_panic(expected = "without a write")]
    fn panics_on_finish_without_write() {
        let hasher = IdentityHasherBuilder.build_hasher();
        let _ = hasher.finish();
    }

    #[test]
    #[should_panic(expected = "16 bytes")]
    fn panics_on_wrong_size_write() {
        let mut hasher = IdentityHasherBuilder.build_hasher();
        hasher.write(&[0u8; 8]);
    }
}
