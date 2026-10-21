//! Where a device's server thread runs: beside the thread that started it.
//!
//! Every launch crosses from the thread that issues it to the device's server thread through
//! shared memory, hundreds of times a step on a decode path. On a CPU whose cores are split
//! between several last-level caches (AMD's chiplets, Intel's tiles), two threads on cores
//! under different caches pay a cross-cache transfer for each of those crossings; on a 6-cache
//! EPYC this made a decode step about 60% slower, depending only on where the scheduler happened
//! to place the two threads. Keeping the server on the cores that share the starting thread's
//! last-level cache keeps the crossings inside one cache.

/// The CPUs sharing a last-level cache with the CPU a thread runs on, among the CPUs the process
/// may run on. Empty where the answer would not narrow anything: one cache over every allowed
/// CPU, a topology the system does not report, or an operating system this is not written for.
pub(crate) struct SharedCacheCpus {
    #[cfg(target_os = "linux")]
    cpus: Option<linux::CpuSet>,
}

impl SharedCacheCpus {
    /// The CPUs sharing the calling thread's last-level cache.
    pub(crate) fn of_current_thread() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            cpus: linux::shared_last_level_cache(),
        }
    }

    /// Keeps the calling thread on these CPUs; a no-op where they are empty or the system
    /// refuses.
    pub(crate) fn keep_current_thread(&self) {
        #[cfg(target_os = "linux")]
        if let Some(cpus) = &self.cpus {
            cpus.keep_current_thread();
        }
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{fs, string::String, vec::Vec};

    /// A set of CPUs as the kernel's affinity calls take it.
    pub(super) struct CpuSet(libc::cpu_set_t);

    // SAFETY: a `cpu_set_t` is a plain bit array.
    unsafe impl Send for CpuSet {}
    unsafe impl Sync for CpuSet {}

    impl CpuSet {
        fn empty() -> Self {
            // SAFETY: an all-zero `cpu_set_t` is the empty set.
            Self(unsafe { core::mem::zeroed() })
        }

        /// The CPUs the calling thread may run on.
        fn allowed() -> Option<Self> {
            let mut set = Self::empty();
            // SAFETY: `set` is a valid `cpu_set_t` of the size passed.
            let read =
                unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set.0) };
            (read == 0).then_some(set)
        }

        fn contains(&self, cpu: usize) -> bool {
            // SAFETY: `CPU_ISSET` reads one bit, and `cpu` is bounds-checked below by the
            // caller's range over `CPU_SETSIZE`.
            unsafe { libc::CPU_ISSET(cpu, &self.0) }
        }

        fn insert(&mut self, cpu: usize) {
            // SAFETY: as in `contains`.
            unsafe { libc::CPU_SET(cpu, &mut self.0) }
        }

        fn len(&self) -> usize {
            // SAFETY: `CPU_COUNT` reads the whole set.
            unsafe { libc::CPU_COUNT(&self.0) as usize }
        }

        pub(super) fn keep_current_thread(&self) {
            // SAFETY: `self.0` is a valid `cpu_set_t` of the size passed. A refusal leaves the
            // thread where it may already run, which is correct, only slower.
            let kept = unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &self.0) };
            if kept != 0 {
                log::debug!("The device server stays unpinned: sched_setaffinity refused");
            }
        }
    }

    /// The allowed CPUs sharing the last-level cache of the CPU the calling thread runs on, or
    /// `None` where that narrows nothing or the topology is not reported.
    pub(super) fn shared_last_level_cache() -> Option<CpuSet> {
        // SAFETY: `sched_getcpu` takes no arguments.
        let cpu = usize::try_from(unsafe { libc::sched_getcpu() }).ok()?;
        let allowed = CpuSet::allowed()?;
        let mut shared = CpuSet::empty();
        for listed in last_level_cache_cpus(cpu)? {
            if listed < libc::CPU_SETSIZE as usize && allowed.contains(listed) {
                shared.insert(listed);
            }
        }
        (shared.len() > 0 && shared.len() < allowed.len()).then_some(shared)
    }

    /// The CPUs sharing `cpu`'s highest-level cache, as sysfs lists them.
    fn last_level_cache_cpus(cpu: usize) -> Option<Vec<usize>> {
        let caches = std::format!("/sys/devices/system/cpu/cpu{cpu}/cache");
        let last = fs::read_dir(&caches)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("index"))
            .filter_map(|entry| {
                let level = read_trimmed(&entry.path().join("level"))?
                    .parse::<u32>()
                    .ok()?;
                Some((level, entry.path()))
            })
            .max_by_key(|(level, _)| *level)?
            .1;
        cpu_list(&read_trimmed(&last.join("shared_cpu_list"))?)
    }

    fn read_trimmed(path: &std::path::Path) -> Option<String> {
        fs::read_to_string(path).ok().map(|text| text.trim().into())
    }

    /// The CPUs a sysfs list names, `0-7,48-55` being sixteen.
    fn cpu_list(text: &str) -> Option<Vec<usize>> {
        let mut cpus = Vec::new();
        for range in text.split(',').filter(|range| !range.is_empty()) {
            match range.split_once('-') {
                Some((first, last)) => {
                    cpus.extend(first.parse::<usize>().ok()?..=last.parse::<usize>().ok()?)
                }
                None => cpus.push(range.parse().ok()?),
            }
        }
        Some(cpus)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_cpu_list_names_its_ranges_and_singles() {
            assert_eq!(
                cpu_list("0-3,8,10-11"),
                Some(std::vec![0, 1, 2, 3, 8, 10, 11])
            );
            assert_eq!(cpu_list("5"), Some(std::vec![5]));
            assert_eq!(cpu_list("0-x"), None);
        }
    }
}
