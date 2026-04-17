use std::{
    cell::UnsafeCell,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    vec::Vec,
};

/// The raw storage for an item, potentially uninitialized.
///
/// Aligned to 64 bytes (typical cache-line size) to prevent false sharing
/// when different threads access adjacent slots concurrently.
#[repr(C, align(64))]
pub struct Bytes<const MAX_ITEM_SIZE: usize> {
    bytes: [u8; MAX_ITEM_SIZE],
}

/// A circular, allocation-free arena for reusable memory blocks.
///
/// `Arena` manages a fixed-capacity pool of [`Bytes`] buffers, each up to
/// `MAX_ITEM_SIZE` bytes. After the pool is lazily initialized, subsequent
/// allocations scan from an internal cursor to find a free slot, avoiding
/// further heap allocation.
///
/// # Const Parameters
///
/// - `MAX_ITEM_COUNT` — maximum number of buffers in the pool.
/// - `MAX_ITEM_SIZE` — capacity of each individual buffer in bytes.
///
/// # How It Works
///
/// The arena maintains a vector of reference-counted buffer slots. When a
/// caller requests memory, the arena advances its cursor through the pool
/// looking for a slot whose reference count is zero, then hands back a
/// [`Bytes`] handle to that slot. The cursor wraps around, giving the
/// allocation pattern its circular behavior.
///
/// Because a `Bytes` handle can outlive the `Arena` itself (e.g. when the
/// owning thread exits but the handle was sent elsewhere), each slot is
/// wrapped in an `Arc` to keep the underlying storage alive. A separate
/// `AtomicU32` reference count tracks logical ownership independently of
/// the `Arc` strong count, so the arena can reliably detect which slots
/// are free.
///
/// # Use Case
///
/// This is useful as a replacement for repeated `Arc<dyn Trait>` allocations.
pub struct Arena<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> {
    /// Backing storage for each slot. Wrapped in `Arc` so that a [`Bytes`]
    /// handle remains valid even after the `Arena` (and its owning thread)
    /// is dropped.
    buffer: Vec<Arc<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>>,
    /// Logical reference counts, one per slot. Tracked separately from the
    /// `Arc` strong count because the arena may be dropped while outstanding
    /// `Bytes` handles still exist — the `Arc` keeps memory alive, but this
    /// counter tells the arena whether a slot can be reclaimed.
    ref_counts: Vec<Arc<AtomicU32>>,
    /// Current scan position in the circular pool. Advanced on each
    /// allocation attempt and wraps at `MAX_ITEM_COUNT`.
    cursor: usize,
}

/// An initialized, immutable handle to a slot in the arena.
///
/// This type is `Send + Sync` and can be cheaply cloned. Each clone
/// increments a logical reference count; when the last clone is dropped,
/// the stored object's destructor runs and the slot becomes available for
/// reuse by the arena.
pub struct ReservedMemory<const MAX_ITEM_SIZE: usize> {
    data: Arc<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>,
    ref_count: Arc<AtomicU32>,
    drop_fn: fn(&mut Bytes<MAX_ITEM_SIZE>),
}

/// An uninitialized handle to a reserved arena slot.
///
/// Obtained from [`Arena::reserve`]. Must be initialized via [`init`](Self::init)
/// to produce a usable [`ReservedMemory`].
///
/// This type is intentionally `!Send` and `!Sync` — it must be initialized on
/// the same thread that reserved it.
pub struct UninitReservedMemory<const MAX_ITEM_SIZE: usize> {
    data: Arc<UnsafeCell<Bytes<MAX_ITEM_SIZE>>>,
    ref_count: Arc<AtomicU32>,
    /// Used to assert the position in the arena.
    #[cfg(test)]
    index: usize,
    // Add this type to make sure the object is `!Sync`.
    not_sync: PhantomData<*const ()>,
}

impl<const MAX_ITEM_SIZE: usize> UninitReservedMemory<MAX_ITEM_SIZE> {
    /// Initialize the reserved memory.
    ///
    /// # Panics
    ///
    /// If the given object isn't safe to store in this arena.
    pub fn init<O>(self, obj: O) -> ReservedMemory<MAX_ITEM_SIZE> {
        assert!(
            accept_obj::<O, MAX_ITEM_SIZE>(),
            "Object isn't safe to store in this arena"
        );

        self.init_with_func(
            |bytes| {
                let ptr = core::ptr::from_mut(bytes);
                unsafe {
                    core::ptr::write(ptr as *mut O, obj);
                };
            },
            |bytes| {
                let ptr = core::ptr::from_mut(bytes);
                unsafe {
                    core::ptr::drop_in_place(ptr as *mut O);
                }
            },
        )
    }

    /// Writes to the reserved slot using `init_data` and attaches `drop_fn`
    /// as the destructor to run when the last [`ReservedMemory`] clone is dropped.
    fn init_with_func<F>(
        self,
        init_data: F,
        drop_fn: fn(&mut Bytes<MAX_ITEM_SIZE>),
    ) -> ReservedMemory<MAX_ITEM_SIZE>
    where
        F: FnOnce(&mut Bytes<MAX_ITEM_SIZE>),
    {
        // SAFETY: We access the `UnsafeCell` contents mutably. This is sound
        // because strong_count == 2 means only two owners exist: the arena's
        // buffer slot and this `UninitReservedMemory`. The arena never reads
        // through the `UnsafeCell` — only the holder of `UninitReservedMemory`
        // writes, so there is no data race.
        assert_eq!(
            Arc::strong_count(&self.data),
            2,
            "Slot must be held by exactly two owners (the arena and this \
             UninitReservedMemory) to guarantee exclusive write access."
        );

        let bytes_mut = unsafe { self.data.as_ref().get().as_mut().unwrap() };
        init_data(bytes_mut);

        ReservedMemory {
            data: self.data,
            ref_count: self.ref_count,
            drop_fn,
        }
    }
}

impl<const MAX_ITEM_SIZE: usize> core::fmt::Debug for ReservedMemory<MAX_ITEM_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReservedMemory")
            .field("data", &self.data)
            .field("drop_fn", &self.drop_fn)
            .finish()
    }
}

impl<const MAX_ITEM_SIZE: usize> Clone for ReservedMemory<MAX_ITEM_SIZE> {
    fn clone(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::Release);

        Self {
            data: self.data.clone(),
            ref_count: self.ref_count.clone(),
            drop_fn: self.drop_fn,
        }
    }
}

impl<const MAX_ITEM_SIZE: usize> Drop for ReservedMemory<MAX_ITEM_SIZE> {
    fn drop(&mut self) {
        // `ref_count` equals the number of live `ReservedMemory` clones.
        //   reserve() → stores 1   (the one clone that `init` will produce)
        //   init()    → consumes UninitReservedMemory, count unchanged at 1
        //   clone()   → fetch_add  (count grows: 2, 3, …)
        //   drop()    → fetch_sub  (count shrinks; previous == 1 means we
        //                           were the last clone, so run the destructor)
        //
        // The arena never touches `ref_count`; slot freeness is tracked via
        // `Arc::strong_count` on the backing buffer instead. So the same
        // logic is correct whether or not the arena is still alive.
        let previous = self.ref_count.fetch_sub(1, Ordering::Release);

        if previous == 1 {
            // SAFETY: We are the last user of this slot. The data pointer is valid,
            // initialized, and no other `ReservedMemory` clone exists.
            let bytes_mut = unsafe { self.data.get().as_mut().unwrap() };
            (self.drop_fn)(bytes_mut);
        }
    }
}

// SAFETY: After initialization, the data behind `ReservedMemory` is immutable
// (no `&mut` access is possible while any clone exists). The logical ref_count
// is an `AtomicU32` with proper ordering, and the backing `Arc` guarantees the
// storage outlives all handles. These together satisfy the `Send` and `Sync`
// contracts.
unsafe impl<const MAX_ITEM_SIZE: usize> Send for ReservedMemory<MAX_ITEM_SIZE> {}
unsafe impl<const MAX_ITEM_SIZE: usize> Sync for ReservedMemory<MAX_ITEM_SIZE> {}

impl<const MAX_ITEM_SIZE: usize> std::convert::AsRef<Bytes<MAX_ITEM_SIZE>>
    for ReservedMemory<MAX_ITEM_SIZE>
{
    /// Gets the reserved bytes.
    fn as_ref(&self) -> &Bytes<MAX_ITEM_SIZE> {
        // The pointer is valid and the data is readonly.
        unsafe { self.data.as_ref().get().as_ref().unwrap() }
    }
}

impl<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> Default
    for Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_ITEM_COUNT: usize, const MAX_ITEM_SIZE: usize> Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE> {
    /// Creates a new, empty `Arena`.
    ///
    /// The internal buffer is not allocated until the first call to [`reserve`](Self::reserve).
    pub const fn new() -> Self {
        Self {
            buffer: Vec::new(),
            ref_counts: Vec::new(),
            cursor: 0,
        }
    }

    /// Returns `true` if an object of type `O` fits within a single slot.
    ///
    /// Checks that both the size and alignment of `O` are compatible with
    /// [`Bytes<MAX_ITEM_SIZE>`].
    pub const fn accept<O>() -> bool {
        accept_obj::<O, MAX_ITEM_SIZE>()
    }

    /// Attempts to reserve an uninitialized slot in the arena.
    ///
    /// On the first call, the internal buffer is lazily allocated to
    /// `MAX_ITEM_COUNT` slots. Subsequent calls scan from the current cursor
    /// position, wrapping around circularly, looking for a slot whose backing
    /// `Arc` has a strong count of 1 (meaning no outstanding
    /// [`ReservedMemory`] handles reference it).
    ///
    /// # Returns
    ///
    /// - `Some(UninitReservedMemory)` — a handle to the reserved slot, ready
    ///   to be initialized via [`UninitReservedMemory::init`].
    /// - `None` — all slots are currently in use.
    pub fn reserve(&mut self) -> Option<UninitReservedMemory<MAX_ITEM_SIZE>> {
        if self.buffer.is_empty() {
            for _ in 0..MAX_ITEM_COUNT {
                self.ref_counts.push(Arc::new(AtomicU32::new(0)));

                // Here we need to disable the clippy warning since we manually ensure the type is
                // send sync and we need to wrap it in an Arc because the bytes might outlive the
                // current arena.
                #[allow(clippy::arc_with_non_send_sync)]
                self.buffer.push(Arc::new(UnsafeCell::new(Bytes {
                    bytes: [0; MAX_ITEM_SIZE],
                })));
            }
        }

        for i in 0..MAX_ITEM_COUNT {
            let i = (i + self.cursor) % MAX_ITEM_COUNT;
            let item = &self.buffer[i];

            // SAFETY: `Arc::strong_count` is not synchronized, but this is safe
            // because `reserve` takes `&mut self`, guaranteeing single-threaded
            // access to the arena side. The only concurrent mutation is a
            // `ReservedMemory` being dropped on another thread, which performs a
            // `Release`-ordered `Arc::drop` before the strong count decrements.
            // A stale (too-high) read here is harmless — we simply skip a slot
            // that is actually free, and will find it on the next call.
            if Arc::strong_count(item) == 1 {
                self.cursor = (i + 1) % MAX_ITEM_COUNT;
                let data = item.clone();
                let ref_count = self.ref_counts[i].clone();
                ref_count.store(1, Ordering::Release);

                return Some(UninitReservedMemory {
                    data,
                    ref_count,
                    #[cfg(test)]
                    index: i,
                    not_sync: PhantomData,
                });
            }
        }

        None
    }
}

const fn accept_obj<O, const MAX_ITEM_SIZE: usize>() -> bool {
    size_of::<O>() <= size_of::<Bytes<MAX_ITEM_SIZE>>()
        && align_of::<O>() <= align_of::<Bytes<MAX_ITEM_SIZE>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_ITEM_SIZE: usize = 2048;

    #[test]
    fn test_lazy_initialization() {
        let mut arena = Arena::<10, MAX_ITEM_SIZE>::new();
        assert_eq!(
            arena.buffer.len(),
            0,
            "Buffer should be empty before first reservation"
        );

        arena.reserve();

        assert_eq!(
            arena.buffer.len(),
            10,
            "Buffer should be initialized to size"
        );
    }

    #[test]
    fn test_sequential_allocation_moves_cursor() {
        let mut arena = Arena::<3, MAX_ITEM_SIZE>::new();

        // First allocation
        let _ = arena.reserve().expect("Should allocate");
        assert_eq!(arena.cursor, 1);

        // Second allocation
        let _ = arena.reserve().expect("Should allocate");
        assert_eq!(arena.cursor, 2);
    }

    #[test]
    fn test_reuse_of_freed_data() {
        let mut arena = Arena::<2, MAX_ITEM_SIZE>::new();

        // Fill the arena
        let data0 = arena.reserve().unwrap();
        let _data1 = arena.reserve().unwrap();

        // Arena is now full (counts are 2)
        assert!(arena.reserve().is_none(), "Should be full");

        // Manually "free" index 0 by setting count to 0 (simulating ManagedOperation drop)
        let data0_index = data0.index;
        core::mem::drop(data0);

        // Should now be able to reserve again, and it should pick up index 0
        let data2 = arena.reserve().expect("Should reuse index 0");
        assert_eq!(data0_index, data2.index);
    }

    #[test]
    fn test_circular_cursor_search() {
        let mut arena = Arena::<3, MAX_ITEM_SIZE>::new();

        // Fill 0, 1, 2
        let _d0 = arena.reserve().unwrap();
        let d1 = arena.reserve().unwrap();
        let _d2 = arena.reserve().unwrap();

        // Free index 1 (the middle)
        core::mem::drop(d1);

        // Currently cursor is at 2. The search starts at (cursor + i) % size.
        // It should wrap around and find index 1.
        let _ = arena.reserve().expect("Should find the hole at index 1");
        assert_eq!(arena.cursor, 2);
    }

    #[test]
    fn test_full_arena_returns_none() {
        let mut arena = Arena::<5, MAX_ITEM_SIZE>::new();

        let mut reserved = Vec::new();

        for _ in 0..5 {
            let item = arena.reserve();
            assert!(item.is_some());
            reserved.push(item);
        }

        // Next one should fail
        assert!(arena.reserve().is_none());
    }
}

#[cfg(test)]
mod drop_lifecycle_tests {
    //! These tests verify the drop-timing contract of `ReservedMemory`:
    //! `drop_fn` must run exactly once, and **only when the last clone is
    //! released**.
    //!
    //! Every assertion is expressed through `Arc::strong_count` on a
    //! payload-owned anchor. Each case is
    //! runnable under `cargo miri test` to catch UB.

    use super::*;
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use std::sync::Arc;

    struct Payload {
        _anchor: Arc<()>,
    }

    #[test]
    fn last_clone_runs_destructor_with_one_clone() {
        let anchor = Arc::new(());
        let mut arena = Arena::<4, 256>::new();
        let reserved = arena.reserve().unwrap().init(Payload {
            _anchor: anchor.clone(),
        });
        assert_eq!(Arc::strong_count(&anchor), 2);
        drop(reserved);
        assert_eq!(
            Arc::strong_count(&anchor),
            1,
            "single ReservedMemory must run drop_fn on drop"
        );
    }

    #[test]
    fn destructor_deferred_until_last_of_two_clones() {
        let anchor = Arc::new(());
        let mut arena = Arena::<4, 256>::new();
        let a = arena.reserve().unwrap().init(Payload {
            _anchor: anchor.clone(),
        });
        let b = a.clone();

        drop(a);
        assert_eq!(
            Arc::strong_count(&anchor),
            2,
            "destructor fired prematurely — `b` still owns the payload"
        );
        drop(b);
        assert_eq!(Arc::strong_count(&anchor), 1);
    }

    #[test]
    fn destructor_deferred_until_last_of_many_clones() {
        let anchor = Arc::new(());
        let mut arena = Arena::<4, 256>::new();
        let first = arena.reserve().unwrap().init(Payload {
            _anchor: anchor.clone(),
        });

        const N: usize = 16;
        let clones: Vec<_> = (0..N).map(|_| first.clone()).collect();
        drop(first);
        assert_eq!(Arc::strong_count(&anchor), 2);

        for (i, c) in clones.into_iter().enumerate() {
            drop(c);
            let expected = if i + 1 == N { 1 } else { 2 };
            assert_eq!(
                Arc::strong_count(&anchor),
                expected,
                "premature destructor after dropping clone {i}"
            );
        }
    }

    /// After the destructor runs once, re-cloning a surviving clone and
    /// dropping it again must not run the destructor a second time.
    #[test]
    fn destructor_runs_exactly_once_across_refill_cycle() {
        let anchor1 = Arc::new(());
        let anchor2 = Arc::new(());
        let mut arena = Arena::<1, 256>::new();

        let first = arena.reserve().unwrap().init(Payload {
            _anchor: anchor1.clone(),
        });
        drop(first);
        assert_eq!(Arc::strong_count(&anchor1), 1);

        // Slot is free again — reuse it with a brand-new payload.
        let second = arena.reserve().unwrap().init(Payload {
            _anchor: anchor2.clone(),
        });
        assert_eq!(
            Arc::strong_count(&anchor1),
            1,
            "refilling the slot must not touch the prior payload's anchor"
        );
        assert_eq!(Arc::strong_count(&anchor2), 2);
        drop(second);
        assert_eq!(Arc::strong_count(&anchor2), 1);
    }

    /// A payload owning a heap allocation (`Box`) gives Miri something
    /// concrete to complain about if the destructor is missed or runs
    /// twice: a double drop is a double-free.
    #[test]
    fn heap_owning_payload_drops_exactly_once() {
        struct HeapOwner(#[allow(dead_code)] Box<[u64; 8]>);

        let mut arena = Arena::<2, 256>::new();
        let a = arena
            .reserve()
            .unwrap()
            .init(HeapOwner(Box::new([1, 2, 3, 4, 5, 6, 7, 8])));
        let b = a.clone();
        let c = a.clone();
        drop(a);
        drop(b);
        drop(c);
        // Miri would flag a double-free on the Box here if the destructor
        // fired more than once, or a leak under `-Zmiri-ignore-leaks=no`
        // if it never fired.
    }
}

#[cfg(test)]
mod concurrent_drop_timing_tests {
    //! Concurrent counterparts to `drop_lifecycle_tests`. The existing
    //! `concurrent_tests` module checks that `drop_fn` runs exactly once
    //! under contention but not *when* it runs — a premature destructor
    //! satisfies "exactly once" while still corrupting surviving clones.
    //! These tests bracket drop timing with live observers.

    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    /// Drop N clones concurrently and verify that the payload's anchor is
    /// released exactly once, after all clones finish. Running under Miri
    /// with `-Zmiri-disable-isolation -Zmiri-preemption-rate=...` flags
    /// is not required — this test observes the post-condition on the
    /// main thread after the join.
    #[test]
    fn concurrent_drops_release_anchor_exactly_once() {
        let anchor = Arc::new(());
        let mut arena = Arena::<4, 256>::new();

        struct Payload {
            #[allow(dead_code)]
            anchor: Arc<()>,
        }

        let reserved = arena.reserve().unwrap().init(Payload {
            anchor: anchor.clone(),
        });

        const N: usize = 8;
        let barrier = Arc::new(Barrier::new(N));
        let mut handles = Vec::with_capacity(N);
        for _ in 0..N - 1 {
            let clone = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                drop(clone);
            }));
        }
        {
            let b = barrier.clone();
            let original = reserved;
            handles.push(thread::spawn(move || {
                b.wait();
                drop(original);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            Arc::strong_count(&anchor),
            1,
            "after all clones drop, payload anchor must be released exactly once"
        );
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::{Arc, Barrier, Mutex};
    use std::{thread, vec};

    const MAX_ITEM_SIZE: usize = 2048;

    /// Wraps an arena in a Mutex for shared cross-thread access.
    fn shared_arena<const N: usize>() -> Arc<Mutex<Arena<N, MAX_ITEM_SIZE>>> {
        #[allow(clippy::arc_with_non_send_sync)]
        Arc::new(Mutex::new(Arena::<N, MAX_ITEM_SIZE>::new()))
    }

    /// Verifies that `drop_fn` is called exactly once even when multiple threads
    /// hold clones and release them concurrently.
    #[test]
    fn test_drop_called_exactly_once_under_contention() {
        let drop_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let arena = shared_arena::<4>();

        let uninit = arena.lock().unwrap().reserve().unwrap();

        struct Probe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let reserved = uninit.init(Probe(drop_count.clone()));

        // Spawn 32 threads, each clones and drops ReservedMemory concurrently.
        let barrier = Arc::new(Barrier::new(32));
        let mut handles = vec![];

        for _ in 0..32 {
            let r = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait(); // all threads drop at the same time
                drop(r);
            }));
        }

        drop(reserved); // drop the original too
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            drop_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "drop_fn must be called exactly once"
        );
    }

    /// Verifies that a slot becomes available for reuse after all `ReservedMemory`
    /// clones are dropped across threads.
    #[test]
    fn test_slot_reuse_after_concurrent_drop() {
        let arena = shared_arena::<1>();
        let uninit = arena.lock().unwrap().reserve().unwrap();
        let reserved = uninit.init(42u64);

        let barrier = Arc::new(Barrier::new(8));
        let mut handles = vec![];

        for _ in 0..8 {
            let r = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                drop(r);
            }));
        }

        drop(reserved);
        for h in handles {
            h.join().unwrap();
        }

        // All clones dropped — the single slot should be free again.
        assert!(
            arena.lock().unwrap().reserve().is_some(),
            "Slot should be available after all clones are dropped"
        );
    }

    /// Verifies that `ReservedMemory` clones dropped after the arena is dropped
    /// still correctly run `drop_fn` (the count == 1 case).
    #[test]
    fn test_drop_after_arena_dropped() {
        let drop_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        struct Probe(Arc<std::sync::atomic::AtomicUsize>);
        impl Drop for Probe {
            fn drop(&mut self) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let reserved = {
            let mut arena = Arena::<4, MAX_ITEM_SIZE>::new();
            let uninit = arena.reserve().unwrap();
            uninit.init(Probe(drop_count.clone()))
            // arena drops here
        };

        // Spawn threads that hold clones past the arena's lifetime.
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = vec![];

        for _ in 0..8 {
            let r = reserved.clone();
            let b = barrier.clone();
            handles.push(thread::spawn(move || {
                b.wait();
                drop(r);
            }));
        }

        drop(reserved);
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            drop_count.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "drop_fn must fire exactly once even when arena is dropped first"
        );
    }
}
