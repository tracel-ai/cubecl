//! Thread affinity: which logical CPUs workers run on, and in what order.

#[cfg(any(target_os = "android", target_os = "linux"))]
mod linux;
#[cfg(any(target_os = "android", target_os = "linux"))]
use linux::Platform;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
use windows::Platform;

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
mod fallback;
#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "windows")))]
use fallback::Platform;

/// A logical CPU, by the number the operating system enumerates it under.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CoreId(usize);

/// A platform's affinity mechanism; the shared ordering policy on top of it
/// is [`get_active_cores`].
trait ThreadAffinity {
    /// The logical CPUs the process may run on, in enumeration order.
    fn active_cpus() -> Vec<CoreId>;

    /// The physical core `cpu` belongs to, represented by its lowest-numbered
    /// SMT sibling, or `None` when the topology cannot be read.
    fn physical_core(cpu: CoreId) -> Option<CoreId>;

    /// Pins the calling thread to `cpu`.
    fn pin_current(cpu: CoreId);
}

/// The logical CPUs workers are pinned to: one per physical core first, SMT
/// siblings after, so a launch with up to `physical cores` units gets a whole
/// core per unit. A CPU whose physical core is unknown counts as its own.
pub fn get_active_cores() -> impl Iterator<Item = CoreId> {
    let cpus = Platform::active_cpus();
    let mut primaries = Vec::with_capacity(cpus.len());
    let mut siblings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cpu in cpus {
        let core = Platform::physical_core(cpu).unwrap_or(cpu);
        if seen.insert(core) {
            primaries.push(cpu);
        } else {
            siblings.push(cpu);
        }
    }
    primaries.into_iter().chain(siblings)
}

/// Pins the calling thread to `core_id`.
pub fn set_for_current(core_id: CoreId) {
    Platform::pin_current(core_id);
}
