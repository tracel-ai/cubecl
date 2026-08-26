//! Thread affinity: which logical CPUs workers run on, and in what order.

#[cfg(any(target_os = "android", target_os = "linux"))]
mod linux;
#[cfg(any(target_os = "android", target_os = "linux"))]
use linux::Platform;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
use windows::Platform;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos::Platform;

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "windows",
    target_os = "macos"
)))]
mod fallback;
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "windows",
    target_os = "macos"
)))]
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

    /// Bytes of a core's L1 data cache, or `None` when it cannot be read.
    fn l1d_cache_size() -> Option<usize>;

    /// Bytes of the last level cache, or `None` when it cannot be read.
    fn llc_cache_size() -> Option<usize>;

    /// Pins the calling thread to `cpu`.
    fn pin_current(cpu: CoreId);
}

/// The logical CPUs workers are pinned to: one per physical core first, SMT
/// siblings after, so a launch with up to `physical cores` units gets a whole
/// core per unit. A CPU whose physical core is unknown counts as its own.
pub fn get_active_cores() -> impl Iterator<Item = CoreId> {
    ordered_cores::<Platform>()
}

/// [`get_active_cores`] over any platform, so the policy is testable on a
/// made-up topology.
fn ordered_cores<P: ThreadAffinity>() -> impl Iterator<Item = CoreId> {
    let cpus = P::active_cpus();
    let mut primaries = Vec::with_capacity(cpus.len());
    let mut siblings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for cpu in cpus {
        let core = P::physical_core(cpu).unwrap_or(cpu);
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

/// Bytes of a core's L1 data cache, or `None` when it cannot be read.
pub fn l1d_cache_size() -> Option<usize> {
    Platform::l1d_cache_size()
}

/// Bytes of the last level cache, or `None` when it cannot be read.
///
/// The size a working set has to outgrow before what it reaches is set by
/// memory rather than by the chip.
pub fn llc_cache_size() -> Option<usize> {
    Platform::llc_cache_size()
}

/// Runs on every platform: the policy on a made-up topology, and the
/// platform's own answers checked for the invariants the policy relies on.
#[cfg(test)]
mod tests {
    use super::*;

    fn ids(cpus: impl IntoIterator<Item = CoreId>) -> Vec<usize> {
        cpus.into_iter().map(|cpu| cpu.0).collect()
    }

    #[test]
    fn cores_are_ordered_one_per_physical_core_then_siblings() {
        // The SMT machine the one running this may not be: sibling pairs
        // (0, 1) and (2, 3), a CPU whose core is unknown, then a lone core.
        struct Fake;

        impl ThreadAffinity for Fake {
            fn active_cpus() -> Vec<CoreId> {
                (0..6).map(CoreId).collect()
            }

            fn physical_core(cpu: CoreId) -> Option<CoreId> {
                match cpu.0 {
                    0 | 1 => Some(CoreId(0)),
                    2 | 3 => Some(CoreId(2)),
                    4 => None,
                    _ => Some(cpu),
                }
            }

            fn l1d_cache_size() -> Option<usize> {
                None
            }

            fn llc_cache_size() -> Option<usize> {
                None
            }

            fn pin_current(_cpu: CoreId) {}
        }

        assert_eq!(ids(ordered_cores::<Fake>()), [0, 2, 4, 5, 1, 3]);
    }

    #[test]
    fn active_cpus_are_distinct_and_ascending() {
        let cpus = ids(Platform::active_cpus());
        assert!(!cpus.is_empty());
        assert!(cpus.windows(2).all(|pair| pair[0] < pair[1]), "{cpus:?}");
    }

    #[test]
    fn active_cores_are_the_active_cpus_reordered() {
        let mut cores = ids(get_active_cores());
        cores.sort_unstable();
        assert_eq!(cores, ids(Platform::active_cpus()));
    }

    #[test]
    fn physical_core_is_a_sibling_no_higher_than_the_cpu_and_its_own_core() {
        for cpu in Platform::active_cpus() {
            let core = Platform::physical_core(cpu);
            assert!(
                core.is_some() || !Platform::READS_TOPOLOGY,
                "no core for {cpu:?}"
            );
            if let Some(core) = core {
                assert!(core <= cpu, "{cpu:?} belongs to {core:?}");
                assert_eq!(Platform::physical_core(core), Some(core));
            }
        }
    }

    #[test]
    fn l1d_cache_size_is_a_plausible_cache_size() {
        let size = Platform::l1d_cache_size();
        assert!(size.is_some() || !Platform::READS_TOPOLOGY);
        if let Some(size) = size {
            assert!((4 * 1024..=1024 * 1024).contains(&size), "{size}");
            assert_eq!(size % 1024, 0, "{size}");
        }
    }

    /// A platform that cannot read the last level says so rather than offering
    /// the deepest cache it does know, which on Apple Silicon would be an
    /// efficiency cluster's L2 six times under the real one. What is reported
    /// has to be a plausible last level, so no smaller than the L1d.
    #[test]
    fn a_reported_last_level_cache_is_a_plausible_one() {
        let Some(llc) = Platform::llc_cache_size() else {
            return;
        };

        assert!(llc <= 4 * 1024 * 1024 * 1024, "{llc}");
        if let Some(l1d) = Platform::l1d_cache_size() {
            assert!(llc >= l1d, "last level {llc} under L1d {l1d}");
        }
    }

    #[test]
    fn pinned_threads_run_on_their_cpu() {
        let cpus = Platform::active_cpus();
        // On a thread of its own: pinning sticks to the thread that asked.
        std::thread::spawn(move || {
            for cpu in cpus {
                set_for_current(cpu);
                // A pinned thread can only be running on its CPU.
                if let Some(running) = Platform::current_cpu() {
                    assert_eq!(running, cpu);
                }
            }
        })
        .join()
        .unwrap();
    }
}
