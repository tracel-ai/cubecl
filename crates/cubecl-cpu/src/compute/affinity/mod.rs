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

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows",)))]
#[inline]
pub fn get_active_cores() -> impl Iterator<Item = CoreId> {
    [].into_iter()
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows",)))]
#[inline]
pub fn set_for_current(_core_id: CoreId) {}
