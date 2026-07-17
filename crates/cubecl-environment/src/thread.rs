/// A unique identifier for a running thread.
///
/// This type is a stub when no std is available to swap with `std::thread::ThreadId`.
#[allow(dead_code)]
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(core::num::NonZeroU64);

#[cfg(multi_threading)]
pub use std::thread::{JoinHandle, spawn};
