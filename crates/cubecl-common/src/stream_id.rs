use crate::stub::{AtomicU64, Ordering};

/// Unique identifier that can represent a stream.
///
/// A stream is not tied one-to-one to a thread anymore: a single execution
/// context can own and switch between several streams. The [`current`](StreamId::current)
/// stream is tracked in ambient state — a thread-local when threads are
/// available, or a single global slot otherwise.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct StreamId {
    /// The value representing the stream id.
    pub value: u64,
}

/// Sentinel stored in the ambient slot before any stream has been assigned.
///
/// The first stream resolved on a context lazily takes id `0` (see
/// [`STREAM_COUNT`]), so the effective default stream is `0` while `u64::MAX`
/// remains free to mean "unset". This makes `u64::MAX` the one value
/// [`StreamId::from_number`] cannot hand out.
const UNSET: u64 = u64::MAX;

/// Monotonic source of implicit stream ids, counting up from `0`.
static STREAM_COUNT: AtomicU64 = AtomicU64::new(0);

#[cfg(multi_threading)]
std::thread_local! {
    static ID: core::cell::Cell<u64> = const { core::cell::Cell::new(UNSET) };
}

/// Single global "current stream" slot for contexts without thread support
/// (`no_std`, or single-threaded targets such as wasm).
#[cfg(not(multi_threading))]
static ID: AtomicU64 = AtomicU64::new(UNSET);

#[cfg(multi_threading)]
fn get_id() -> u64 {
    ID.with(|id| id.get())
}

#[cfg(multi_threading)]
fn set_id(value: u64) {
    ID.with(|id| id.set(value));
}

#[cfg(not(multi_threading))]
fn get_id() -> u64 {
    ID.load(Ordering::Relaxed)
}

#[cfg(not(multi_threading))]
fn set_id(value: u64) {
    ID.store(value, Ordering::Relaxed);
}

/// Resolve the current stream id, lazily assigning a fresh one on first use.
fn current_value() -> u64 {
    let current = get_id();
    if current == UNSET {
        let new = STREAM_COUNT.fetch_add(1, Ordering::Relaxed);
        set_id(new);
        new
    } else {
        current
    }
}

impl StreamId {
    /// Executes `f` on this stream, restoring the previous stream afterward.
    ///
    /// The previous [`StreamId`] is saved before the call and restored on
    /// return — including on unwind — so the caller never has to manage
    /// raw `swap` pairs.
    pub fn executes<F, T>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        struct Guard(StreamId);

        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe {
                    StreamId::swap(self.0);
                }
            }
        }

        let old = unsafe { StreamId::swap(self) };
        let guard = Guard(old);

        let returned = f();
        core::mem::drop(guard);
        returned
    }

    /// Get the current stream id.
    pub fn current() -> Self {
        Self {
            value: current_value(),
        }
    }

    /// Allocate a fresh, automatically-assigned stream id.
    ///
    /// Unlike [`current`](Self::current), this always mints a brand new id from
    /// the global counter instead of returning the ambient one. Use it to give
    /// an explicitly created stream its own identity up front.
    pub fn fresh() -> Self {
        Self {
            value: STREAM_COUNT.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Build a stream id from a user-chosen `number`.
    ///
    /// The number is the id: `from_number(5)` is `StreamId { value: 5 }`. The
    /// same number therefore always maps to the same stream, so callers can
    /// deliberately pin work to a shared stream — and thus a shared memory
    /// pool — across threads.
    ///
    /// # Collisions
    ///
    /// User numbers share one id space with the implicit ids handed out by
    /// [`current`](Self::current)/[`fresh`](Self::fresh), which count up from
    /// `0`. Asking for number `5` gives you the same stream a thread that
    /// lazily took implicit id `5` is already on. This is deliberate: a chosen
    /// number is honoured exactly as given, and avoiding overlap is the
    /// caller's job — pick numbers above the count of streams you expect to be
    /// created implicitly, or spawn every stream explicitly.
    ///
    /// Note that overlap is possible even between distinct ids: runtimes map
    /// ids onto a fixed pool with `value % max_streams`, so ids congruent
    /// modulo that limit share a physical stream regardless.
    ///
    /// # Panics
    ///
    /// In debug builds, on `u64::MAX` — that value is reserved as the internal
    /// "no stream assigned yet" sentinel. In release builds it silently reads
    /// back as unassigned.
    pub fn from_number(number: u64) -> Self {
        debug_assert_ne!(
            number, UNSET,
            "stream number {number} is reserved as the `UNSET` sentinel"
        );
        Self { value: number }
    }

    /// Swap the current stream id for the given one, returning the previous one.
    ///
    /// # Safety
    ///
    /// Unknown at this point, don't use that if you don't know what you are doing.
    pub unsafe fn swap(stream: StreamId) -> StreamId {
        let old = Self::current();
        set_id(stream.value);
        old
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StreamId({:?})", self.value))
    }
}
