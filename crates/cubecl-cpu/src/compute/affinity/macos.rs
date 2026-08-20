use std::ffi::{CStr, c_void};
use std::{mem, ptr};

use super::{CoreId, ThreadAffinity};
use libc::{
    THREAD_AFFINITY_POLICY, THREAD_AFFINITY_POLICY_COUNT, pthread_mach_thread_np, pthread_self,
    sysctlbyname, thread_affinity_policy_data_t, thread_policy_set,
};

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    fn active_cpus() -> Vec<CoreId> {
        // Darwin has no per-process affinity mask, so every online CPU is
        // available.
        let cpus = sysctl(c"hw.logicalcpu")
            .map(|n| n as usize)
            .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
            .unwrap_or(1);
        (0..cpus).map(CoreId).collect()
    }

    fn physical_core(cpu: CoreId) -> Option<CoreId> {
        // Darwin numbers a core's SMT siblings consecutively, so with N
        // threads per core, CPU k belongs to the core starting at k - k % N.
        // Apple Silicon has no SMT, making every CPU its own core.
        let logical = sysctl(c"hw.logicalcpu")? as usize;
        let physical = sysctl(c"hw.physicalcpu")? as usize;
        let per_core = (logical / physical.max(1)).max(1);
        Some(CoreId(cpu.0 - cpu.0 % per_core))
    }

    fn l1d_cache_size() -> Option<usize> {
        // On heterogeneous Apple Silicon this reports the efficiency cores'
        // (smallest) L1d. Threads cannot be pinned there, so any worker may
        // land on an efficiency core and the smallest cache is the safe cap.
        sysctl(c"hw.l1dcachesize").map(|bytes| bytes as usize)
    }

    fn pin_current(cpu: CoreId) {
        // Darwin has no hard pinning; the affinity tag only hints that
        // threads with different tags prefer different L2 caches, and Apple
        // Silicon rejects even that with KERN_NOT_SUPPORTED. Tag 0 is
        // THREAD_AFFINITY_TAG_NULL, so offset by one.
        let mut policy = thread_affinity_policy_data_t {
            affinity_tag: cpu.0 as libc::integer_t + 1,
        };
        unsafe {
            thread_policy_set(
                pthread_mach_thread_np(pthread_self()),
                THREAD_AFFINITY_POLICY as libc::thread_policy_flavor_t,
                &mut policy as *mut _ as libc::thread_policy_t,
                THREAD_AFFINITY_POLICY_COUNT,
            );
        }
    }
}

/// Reads an integer sysctl, accepting the 32- and 64-bit widths the `hw`
/// values come in.
fn sysctl(name: &CStr) -> Option<u64> {
    let mut value: u64 = 0;
    let mut size = mem::size_of::<u64>();
    let ret = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut value as *mut _ as *mut c_void,
            &mut size,
            ptr::null_mut(),
            0,
        )
    };
    // A 4-byte value lands in the low bytes of the zeroed buffer on the
    // little-endian targets macOS runs on.
    (ret == 0 && (size == 4 || size == 8)).then_some(value)
}
