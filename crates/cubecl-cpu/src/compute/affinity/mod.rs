#[cfg(any(target_os = "android", target_os = "linux"))]
mod linux;
#[cfg(any(target_os = "android", target_os = "linux"))]
pub use linux::get_active_cores;
#[cfg(any(target_os = "android", target_os = "linux"))]
pub use linux::set_for_current;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::get_active_cores;
#[cfg(target_os = "windows")]
pub use windows::set_for_current;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreId(usize);

/// Platforms without an affinity API (e.g. macOS) still report their cores — the
/// threadpool sizes itself off this iterator, and zero cores means zero workers (the
/// dispatcher then panics on its empty queue list and any launch hangs). Pinning stays
/// a no-op.
#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows",)))]
#[inline]
pub fn get_active_cores() -> impl Iterator<Item = CoreId> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    (0..cores).map(CoreId)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows",)))]
#[inline]
pub fn set_for_current(_core_id: CoreId) {}
