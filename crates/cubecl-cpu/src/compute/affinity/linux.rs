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

    fn pin_current(cpu: CoreId) {
        let mut set = new_cpu_set();
        let tid = unsafe { syscall(SYS_gettid) } as libc::id_t;
        unsafe { libc::setpriority(libc::PRIO_PROCESS, tid, 0) };
        unsafe { CPU_SET(cpu.0, &mut set) };
        unsafe { sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &set) };
    }
}

fn get_affinity_mask() -> cpu_set_t {
    let mut set = new_cpu_set();
    unsafe { sched_getaffinity(0, mem::size_of::<cpu_set_t>(), &mut set) };
    set
}

fn new_cpu_set() -> cpu_set_t {
    unsafe { std::mem::zeroed::<cpu_set_t>() }
}
