use std::mem;

use super::{CoreId, ThreadAffinity};
use libc::{
    CPU_ISSET, CPU_SET, CPU_SETSIZE, SYS_gettid, cpu_set_t, sched_getaffinity, sched_setaffinity,
    syscall,
};

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        let affinity_mask = get_affinity_mask();
        (0..CPU_SETSIZE as usize)
            .filter(|i| unsafe { CPU_ISSET(*i, &affinity_mask) })
            .map(CoreId)
            .collect()
    }

    fn physical_core(cpu: CoreId) -> Option<CoreId> {
        // libc exposes only the affinity mask; the kernel publishes SMT
        // topology solely through sysfs, its stable ABI for it. The siblings
        // list is smallest-first, so the first entry is the same for every
        // sibling of the core and serves as its identity.
        let list = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
            cpu.0
        ))
        .ok()?;
        let first = list.split(['-', ',']).next()?.trim().parse().ok()?;
        Some(CoreId(first))
    }

    fn l1d_cache_size() -> Option<usize> {
        (0..=5).find_map(|index| {
            let cache = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
            let read = |file: &str| std::fs::read_to_string(format!("{cache}/{file}")).ok();
            (read("level")?.trim() == "1").then_some(())?;
            (read("type")?.trim() == "Data").then_some(())?;
            parse_cache_size(read("size")?.trim())
        })
    }

    fn llc_cache_size() -> Option<usize> {
        // The deepest level sysfs reports, which is the cache a working set has
        // to outgrow before it is streaming from memory rather than from chip.
        (0..=5)
            .filter_map(|index| {
                let cache = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
                let read = |file: &str| std::fs::read_to_string(format!("{cache}/{file}")).ok();
                let level: u32 = read("level")?.trim().parse().ok()?;
                (read("type")?.trim() != "Instruction").then_some(())?;
                Some((level, parse_cache_size(read("size")?.trim())?))
            })
            .max_by_key(|(level, _)| *level)
            .map(|(_, size)| size)
    }

    fn pin_current(cpu: CoreId) {
        let mut set = new_cpu_set();
        let tid = unsafe { syscall(SYS_gettid) } as libc::id_t;
        unsafe { libc::setpriority(libc::PRIO_PROCESS, tid, 0) };
        unsafe { CPU_SET(cpu.0, &mut set) };
        unsafe { sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &set) };
    }
}

/// Parses sysfs cache sizes, e.g. `"1024K"` or `"32M"`.
fn parse_cache_size(size: &str) -> Option<usize> {
    let (number, unit) = size.split_at(size.len().checked_sub(1)?);
    let scale = match unit {
        "K" => 1024,
        "M" => 1024 * 1024,
        _ => return None,
    };
    Some(number.parse::<usize>().ok()? * scale)
}

fn get_affinity_mask() -> cpu_set_t {
    let mut set = new_cpu_set();
    unsafe { sched_getaffinity(0, mem::size_of::<cpu_set_t>(), &mut set) };
    set
}

fn new_cpu_set() -> cpu_set_t {
    unsafe { std::mem::zeroed::<cpu_set_t>() }
}

/// What the tests in [`super`] need of this platform beyond the trait.
#[cfg(test)]
impl Platform {
    /// sysfs reports a real topology here.
    pub(super) const READS_TOPOLOGY: bool = true;

    /// The CPU the calling thread is on, which a pinned thread cannot leave.
    pub(super) fn current_cpu() -> Option<CoreId> {
        let cpu = unsafe { libc::sched_getcpu() };
        usize::try_from(cpu).ok().map(CoreId)
    }
}
