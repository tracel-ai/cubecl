use std::mem;

use super::CoreId;
use libc::{
    CPU_ISSET, CPU_SET, CPU_SETSIZE, SYS_gettid, cpu_set_t, sched_getaffinity, sched_setaffinity,
    syscall,
};

pub fn get_active_cores() -> impl Iterator<Item = CoreId> {
    let affinity_mask = get_affinity_mask();

    (0..CPU_SETSIZE as usize)
        .into_iter()
        .filter(move |i| unsafe { CPU_ISSET(*i, &affinity_mask) })
        .map(|i| CoreId(i))
}

pub fn set_for_current(core_id: CoreId) {
    let mut set = new_cpu_set();
    let tid = unsafe { syscall(SYS_gettid) } as libc::id_t;
    unsafe { libc::setpriority(libc::PRIO_PROCESS, tid, 0) };
    unsafe { CPU_SET(core_id.0, &mut set) };
    unsafe { sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &set) };
}

fn get_affinity_mask() -> cpu_set_t {
    let mut set = new_cpu_set();
    unsafe { sched_getaffinity(0, mem::size_of::<cpu_set_t>(), &mut set) };
    set
}

fn new_cpu_set() -> cpu_set_t {
    unsafe { std::mem::zeroed::<cpu_set_t>() }
}
