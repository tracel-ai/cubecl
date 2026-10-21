//! Where a device's server thread runs: under the same last-level cache as the thread issuing
//! its launches.
//!
//! Every launch crosses from the thread that issues it to the device's server thread through
//! shared memory, hundreds of times a step on a decode path. On a CPU whose cores are split
//! between several last-level caches (AMD's chiplets, Intel's tiles), two threads under
//! different caches pay a cross-cache transfer on each of those crossings, and a decode step
//! ran about 60% slower when the scheduler happened to place them apart. The scheduler may
//! also move the issuing thread at any time, so the server follows it: the issuing thread
//! states the CPU it runs on each time it hands over a full buffer of tasks
//! ([`current_cpu`]), and the server moves to that CPU's cache once the issuer has settled
//! there ([`ServerPlacement::follow`]).

/// The CPU the calling thread runs on; [`UNKNOWN_CPU`] where the system does not say.
pub(crate) fn current_cpu() -> u32 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `sched_getcpu` takes no arguments.
        u32::try_from(unsafe { libc::sched_getcpu() }).unwrap_or(UNKNOWN_CPU)
    }
    #[cfg(not(target_os = "linux"))]
    {
        UNKNOWN_CPU
    }
}

/// What [`current_cpu`] answers where the system does not say.
pub(crate) const UNKNOWN_CPU: u32 = u32::MAX;

/// Buffers in a row an issuer fills from one other last-level cache before the server moves
/// there. Several threads may issue a device's launches from different caches, each now and
/// then; moving on every buffer would chase each of them in turn, a migration every time.
const SETTLED_BUFFERS: u32 = 8;

/// The last-level cache a device's server thread is kept under, and where the issuer has been
/// filling its buffers from instead, if anywhere.
#[derive(Default)]
pub(crate) struct ServerPlacement {
    /// The cache domain the server is kept under; `None` until the first buffer.
    kept: Option<usize>,
    /// A domain other than `kept` the last buffers came from, and how many in a row.
    elsewhere: Option<(usize, u32)>,
}

impl ServerPlacement {
    /// Keeps the calling thread, the server, under `issuer_cpu`'s last-level cache: at once for
    /// the first buffer, and once [`SETTLED_BUFFERS`] in a row came from a cache other than the
    /// one it is kept under after that. A no-op on a CPU with one last-level cache over the
    /// CPUs the process may use, for an unknown CPU, and on systems this is not written for.
    pub(crate) fn follow(&mut self, issuer_cpu: u32) {
        #[cfg(target_os = "linux")]
        {
            let Some(domains) = linux::CacheDomains::get() else {
                return;
            };
            if let Some(domain) = domains
                .of_cpu(issuer_cpu)
                .and_then(|domain| self.settle(domain))
            {
                domains.keep_current_thread(domain);
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = issuer_cpu;
    }

    /// Notes that a buffer came from `domain`, and answers the domain the server moves to, if
    /// it moves.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn settle(&mut self, domain: usize) -> Option<usize> {
        let moves = match (self.kept, self.elsewhere) {
            (None, _) => true,
            (Some(kept), _) if kept == domain => {
                self.elsewhere = None;
                false
            }
            (Some(_), Some((other, buffers))) if other == domain => {
                self.elsewhere = Some((domain, buffers + 1));
                buffers + 1 >= SETTLED_BUFFERS
            }
            (Some(_), _) => {
                self.elsewhere = Some((domain, 1));
                SETTLED_BUFFERS <= 1
            }
        };
        moves.then(|| {
            self.kept = Some(domain);
            self.elsewhere = None;
            domain
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_buffer_places_the_server() {
        assert_eq!(ServerPlacement::default().settle(3), Some(3));
    }

    #[test]
    fn issuers_taking_turns_from_two_caches_leave_the_server_where_it_is() {
        let mut placement = ServerPlacement::default();
        placement.settle(1);
        for _ in 0..100 {
            assert_eq!(placement.settle(3), None);
            assert_eq!(placement.settle(1), None);
        }
    }

    #[test]
    fn an_issuer_settled_under_another_cache_draws_the_server_there() {
        let mut placement = ServerPlacement::default();
        placement.settle(1);
        for _ in 1..SETTLED_BUFFERS {
            assert_eq!(placement.settle(3), None);
        }
        assert_eq!(placement.settle(3), Some(3));
        assert_eq!(placement.settle(3), None);
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{fs, path::Path, string::String, sync::OnceLock, vec::Vec};

    /// The CPUs the process may run on, grouped by the last-level cache they share.
    pub(super) struct CacheDomains {
        /// Each CPU's domain, by CPU number; `None` for a CPU the process may not run on.
        of_cpu: Vec<Option<usize>>,
        /// Each domain's CPUs.
        cpus: Vec<CpuSet>,
    }

    impl CacheDomains {
        /// The domains, read once; `None` where there is only one or the system does not say.
        ///
        /// Read on the first call, before any thread is moved: the allowed CPUs are the
        /// process's, not those of a server already kept under one cache.
        pub(super) fn get() -> Option<&'static Self> {
            static DOMAINS: OnceLock<Option<CacheDomains>> = OnceLock::new();
            DOMAINS.get_or_init(Self::read).as_ref()
        }

        fn read() -> Option<Self> {
            let allowed = CpuSet::allowed()?;
            let mut of_cpu = Vec::new();
            let mut cpus: Vec<CpuSet> = Vec::new();
            // Each domain's lowest CPU, which names it in sysfs's lists.
            let mut lowest_cpus: Vec<usize> = Vec::new();
            for cpu in (0..libc::CPU_SETSIZE as usize).filter(|&cpu| allowed.contains(cpu)) {
                let lowest = *last_level_cache_cpus(cpu)?.iter().min()?;
                let domain = match lowest_cpus.iter().position(|&known| known == lowest) {
                    Some(domain) => domain,
                    None => {
                        lowest_cpus.push(lowest);
                        cpus.push(CpuSet::empty());
                        cpus.len() - 1
                    }
                };
                cpus[domain].insert(cpu);
                of_cpu.resize(of_cpu.len().max(cpu + 1), None);
                of_cpu[cpu] = Some(domain);
            }
            (cpus.len() > 1).then_some(Self { of_cpu, cpus })
        }

        /// The domain of `cpu`, if the process may run on it.
        pub(super) fn of_cpu(&self, cpu: u32) -> Option<usize> {
            *self.of_cpu.get(cpu as usize)?
        }

        /// Keeps the calling thread on `domain`'s CPUs. A refusal leaves the thread where it
        /// may already run, which is correct, only slower.
        pub(super) fn keep_current_thread(&self, domain: usize) {
            let set = &self.cpus[domain].0;
            // SAFETY: `set` is a valid `cpu_set_t` of the size passed.
            let kept = unsafe { libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), set) };
            if kept != 0 {
                log::debug!("The device server stays where it is: sched_setaffinity refused");
            }
        }
    }

    /// A set of CPUs as the kernel's affinity calls take it.
    struct CpuSet(libc::cpu_set_t);

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

        /// Whether `cpu`, below `CPU_SETSIZE`, is in the set.
        fn contains(&self, cpu: usize) -> bool {
            // SAFETY: `CPU_ISSET` reads one bit of a valid set.
            unsafe { libc::CPU_ISSET(cpu, &self.0) }
        }

        /// Adds `cpu`, below `CPU_SETSIZE`.
        fn insert(&mut self, cpu: usize) {
            // SAFETY: `CPU_SET` writes one bit of a valid set.
            unsafe { libc::CPU_SET(cpu, &mut self.0) }
        }
    }

    // SAFETY: a `cpu_set_t` is a plain bit array.
    unsafe impl Send for CpuSet {}
    unsafe impl Sync for CpuSet {}

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

    fn read_trimmed(path: &Path) -> Option<String> {
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

        #[test]
        fn a_server_that_follows_a_cpu_is_kept_under_its_cache() {
            let Some(domains) = CacheDomains::get() else {
                return;
            };
            std::thread::spawn(move || {
                let cpu = super::super::current_cpu();
                super::super::ServerPlacement::default().follow(cpu);
                let domain = domains.of_cpu(cpu).unwrap();
                let kept = CpuSet::allowed().unwrap();
                for other in 0..libc::CPU_SETSIZE as usize {
                    assert_eq!(kept.contains(other), domains.cpus[domain].contains(other));
                }
            })
            .join()
            .unwrap();
        }

        #[test]
        fn every_allowed_cpu_lies_in_its_own_domain() {
            let Some(domains) = CacheDomains::get() else {
                return;
            };
            let allowed = CpuSet::allowed().unwrap();
            for cpu in (0..libc::CPU_SETSIZE as usize).filter(|&cpu| allowed.contains(cpu)) {
                let domain = domains.of_cpu(cpu as u32).unwrap();
                assert!(domains.cpus[domain].contains(cpu));
            }
        }
    }
}
