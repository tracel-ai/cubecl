#[cfg(not(feature = "std"))]
use spin::{Mutex as MutexImported, MutexGuard, Once as OnceImported, RwLock as RwLockImported};
#[cfg(feature = "std")]
use std::sync::{
    Mutex as MutexImported, MutexGuard, OnceLock as OnceImported, RwLock as RwLockImported,
};

#[cfg(not(feature = "std"))]
pub use spin::{Lazy, RwLockReadGuard, RwLockWriteGuard};
#[cfg(feature = "std")]
pub use std::sync::{LazyLock as Lazy, RwLockReadGuard, RwLockWriteGuard};

/// A spin-based one-time initialization cell, identical on every target.
///
/// Prefer [`SyncOnceCell`] for plain lazy initialization; use this when the
/// fallible [`spin::Once::try_call_once`] API is needed.
pub use spin::Once;

#[cfg(target_has_atomic = "ptr")]
pub use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
pub use portable_atomic_util::Arc;

/// A mutual exclusion primitive useful for protecting shared data
///
/// This mutex will block threads waiting for the lock to become available. The
/// mutex can also be statically initialized or created via a [`Mutex::new`]
///
/// [Mutex] wrapper to make `spin::Mutex` API compatible with `std::sync::Mutex` to swap
#[derive(Debug, Default)]
pub struct Mutex<T> {
    inner: MutexImported<T>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: MutexImported::new(value),
        }
    }

    /// Locks the mutex blocking the current thread until it is able to do so.
    ///
    /// A poisoned lock is recovered rather than reported: a panic under a
    /// guard must not turn every later lock into a panic, matching the
    /// unwind behavior of the spin implementation used off-std. The `Result`
    /// is kept for API stability and is always `Ok`.
    #[inline(always)]
    pub fn lock(&self) -> Result<MutexGuard<'_, T>, alloc::string::String> {
        #[cfg(not(feature = "std"))]
        {
            Ok(self.inner.lock())
        }

        #[cfg(feature = "std")]
        {
            Ok(self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()))
        }
    }
}

/// A reader-writer lock which is exclusively locked for writing or shared for reading.
/// This reader-writer lock will block threads waiting for the lock to become available.
/// The lock can also be statically initialized or created via a [`RwLock::new`]
/// [`RwLock`] wrapper to make `spin::RwLock` API compatible with `std::sync::RwLock` to swap
#[derive(Debug)]
pub struct RwLock<T> {
    inner: RwLockImported<T>,
}

impl<T> RwLock<T> {
    /// Creates a new reader-writer lock in an unlocked state ready for use.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self {
            inner: RwLockImported::new(value),
        }
    }

    /// Locks this rwlock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// Poisoning is recovered, never reported; see [`Mutex::lock`].
    #[inline(always)]
    pub fn read(&self) -> Result<RwLockReadGuard<'_, T>, alloc::string::String> {
        #[cfg(not(feature = "std"))]
        {
            Ok(self.inner.read())
        }
        #[cfg(feature = "std")]
        {
            Ok(self
                .inner
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner()))
        }
    }

    /// Locks this rwlock with exclusive write access, blocking the current thread
    /// until it can be acquired.
    ///
    /// Poisoning is recovered, never reported; see [`Mutex::lock`].
    #[inline(always)]
    pub fn write(&self) -> Result<RwLockWriteGuard<'_, T>, alloc::string::String> {
        #[cfg(not(feature = "std"))]
        {
            Ok(self.inner.write())
        }

        #[cfg(feature = "std")]
        {
            Ok(self
                .inner
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()))
        }
    }
}

/// A cell that provides lazy one-time initialization that implements [Sync] and [Send].
///
/// This module is a stub when no std is available to swap with [`std::sync::OnceLock`].
pub struct SyncOnceCell<T>(OnceImported<T>);

impl<T: core::fmt::Debug> Default for SyncOnceCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: core::fmt::Debug> SyncOnceCell<T> {
    /// Create a new once.
    #[inline(always)]
    pub fn new() -> Self {
        Self(OnceImported::new())
    }

    /// Initialize the cell with a value.
    #[inline(always)]
    pub fn initialized(value: T) -> Self {
        #[cfg(not(feature = "std"))]
        {
            let cell = OnceImported::initialized(value);
            Self(cell)
        }

        #[cfg(feature = "std")]
        {
            let cell = OnceImported::new();
            cell.set(value).unwrap();

            Self(cell)
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell
    /// was empty.
    #[inline(always)]
    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        #[cfg(not(feature = "std"))]
        {
            self.0.call_once(f)
        }

        #[cfg(feature = "std")]
        {
            self.0.get_or_init(f)
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Regression: a panic under a guard must not poison the lock — one
    /// failed autotune must not take down every later kernel launch.
    #[test]
    fn poisoned_mutex_recovers() {
        let mutex = Mutex::new(0u32);

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mutex.lock().unwrap();
            panic!("poison the lock");
        }))
        .unwrap_err();

        *mutex.lock().unwrap() += 1;
        assert_eq!(*mutex.lock().unwrap(), 1);
    }

    #[test]
    fn poisoned_rwlock_recovers() {
        let lock = RwLock::new(0u32);

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = lock.write().unwrap();
            panic!("poison the lock");
        }))
        .unwrap_err();

        assert_eq!(*lock.read().unwrap(), 0);
    }
}
