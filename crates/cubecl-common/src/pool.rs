//! A dynamically-growing pool of single-cell items handed out as exclusive, non-cloneable handles.

use crate::stub::{AtomicBool, Ordering};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;

/// Resets a pooled value so the pool can hand it out again.
pub trait Reclaim {
    /// Clear the value in place, keeping its allocation for the next acquire.
    fn reclaim(&mut self);
}

/// A pooled value that stays owned by the [`LeasePool`] even while checked out.
///
/// `leased` is the free/used flag; the value lives in an `UnsafeCell` so a handout can mutate it in
/// place while the pool still holds a shared reference to the item.
struct LeaseSlot<T: Default + Reclaim> {
    leased: AtomicBool,
    item: UnsafeCell<T>,
}

impl<T: Default + Reclaim> Default for LeaseSlot<T> {
    fn default() -> Self {
        Self {
            leased: AtomicBool::new(false),
            item: UnsafeCell::new(T::default()),
        }
    }
}

// SAFETY: `leased` guarantees at most one live `LeaseHandle` references an item at a
// time, so writes through the `UnsafeCell` are never aliased by another writer, and the pool only
// ever forms shared references to the item (to read/write `leased`).
unsafe impl<T: Default + Reclaim> Send for LeaseSlot<T> {}
unsafe impl<T: Default + Reclaim> Sync for LeaseSlot<T> {}

/// Pool of reusable values with single-cell items that are not cloneable.
///
/// Items stay in the pool while checked out; [`Self::acquire`] hands back a
/// [`LeaseHandle`] pointing at one and flags it `leased`. Dropping the handle reclaims the
/// value and frees the slot, so allocations are reused across acquires instead of allocating anew.
#[derive(Default)]
pub struct LeasePool<T: Default + Reclaim> {
    /// Boxed so each item keeps a stable address when the `Vec` grows — handles hold raw pointers
    /// into these allocations, which must stay valid across `push`.
    items: Vec<Box<LeaseSlot<T>>>,
}

impl<T: Default + Reclaim> core::fmt::Debug for LeasePool<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LeasePool")
            .field("items", &self.items.len())
            .finish()
    }
}

impl<T: Default + Reclaim> LeasePool<T> {
    /// Pre-reserve room for `capacity` pooled items (if exceeded, the pool still grows on demand).
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Take an item, reusing a free pooled item when available.
    pub fn acquire(&mut self) -> LeaseHandle<T> {
        let item: &LeaseSlot<T> = match self.items.iter().find(|item| {
            item.leased
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        }) {
            Some(item) => item,
            None => {
                let item = Box::new(LeaseSlot::default());
                item.leased.store(true, Ordering::Release);
                self.items.push(item);
                self.items.last().unwrap()
            }
        };

        LeaseHandle { item }
    }
}

/// Handle to a pooled item. Reclaims the value and frees its slot when dropped.
///
/// The pointee is owned by the [`LeasePool`], which never removes items, so it outlives
/// every handle. An atomic `leased` flag keeps the reference unique for the handle's lifetime.
pub struct LeaseHandle<T: Default + Reclaim> {
    item: *const LeaseSlot<T>,
}

// SAFETY: the handle may be moved across threads by its holder. The pointee is `Send + Sync` and,
// per the `leased` flag invariant, uniquely owned by this handle.
unsafe impl<T: Default + Reclaim> Send for LeaseHandle<T> {}

impl<T: Default + Reclaim> core::ops::Deref for LeaseHandle<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: unique access is guaranteed by the `leased` flag; the pointee outlives the handle.
        unsafe { &*(*self.item).item.get() }
    }
}

impl<T: Default + Reclaim> core::ops::DerefMut for LeaseHandle<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see `deref`; `&mut self` guarantees no other reference through this handle.
        unsafe { &mut *(*self.item).item.get() }
    }
}

impl<T: Default + Reclaim> core::fmt::Debug for LeaseHandle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LeaseHandle").finish_non_exhaustive()
    }
}

impl<T: Default + Reclaim> Drop for LeaseHandle<T> {
    fn drop(&mut self) {
        // SAFETY: the pool outlives the handle, so the pointee is still valid.
        let item = unsafe { &*self.item };
        // Reclaim any value not drained by the holder (e.g. a caller that failed mid-use), keeping
        // the allocation for the next acquire.
        unsafe { &mut *item.item.get() }.reclaim();
        item.leased.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    /// Stand-in pooled value: a `Vec` so we can observe capacity reuse and reclaim clearing.
    #[derive(Default)]
    struct Probe(Vec<u32>);

    impl Reclaim for Probe {
        fn reclaim(&mut self) {
            self.0.clear();
        }
    }

    impl<T: Default + Reclaim> LeasePool<T> {
        fn len(&self) -> usize {
            self.items.len()
        }
    }

    /// The key soundness case: acquiring more items grows `items`, which must not invalidate
    /// handles already handed out (they hold raw pointers into the boxed slots). Run under Miri
    /// to check the aliasing model, not just the observable values.
    #[test]
    fn growth_keeps_live_handles_valid() {
        // Start empty so every acquire pushes a new item, reallocating `items` while earlier
        // handles are still live.
        let mut pool = LeasePool::<Probe>::with_capacity(0);

        let mut a = pool.acquire();
        a.0.push(1);
        let mut b = pool.acquire();
        b.0.push(2);
        let mut c = pool.acquire();
        c.0.push(3);

        // Write through the earliest handle again, after two growth-triggering pushes.
        a.0.push(10);

        assert_eq!(a.0, vec![1, 10]);
        assert_eq!(b.0, vec![2]);
        assert_eq!(c.0, vec![3]);
    }

    /// Dropping a handle reclaims (clears) the value but keeps its allocation for the next acquire.
    #[test]
    fn drop_reclaims_and_reuses_allocation() {
        let mut pool = LeasePool::<Probe>::with_capacity(0);

        let mut a = pool.acquire();
        a.0.extend([1, 2, 3, 4]);
        let cap = a.0.capacity();
        assert!(cap >= 4);
        drop(a);

        // The only free slot is the one just released; it comes back cleared but with its
        // allocation intact.
        let reused = pool.acquire();
        assert!(reused.0.is_empty());
        assert_eq!(reused.0.capacity(), cap);
    }

    /// A released slot is reused on the next acquire rather than growing the pool.
    #[test]
    fn frees_slot_for_later_acquire() {
        let mut pool = LeasePool::<Probe>::with_capacity(0);
        for _ in 0..8 {
            let mut handle = pool.acquire();
            handle.0.push(0);
        }
        // Only ever one live handle at a time, so the pool holds exactly one item.
        assert_eq!(pool.len(), 1);
    }
}
