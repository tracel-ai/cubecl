use winapi::shared::basetsd::{DWORD_PTR, PDWORD_PTR};
use winapi::um::processthreadsapi::{GetCurrentProcess, GetCurrentThread};
use winapi::um::winbase::{GetProcessAffinityMask, SetThreadAffinityMask};

use super::{CoreId, ThreadAffinity};

pub(super) struct Platform;

impl ThreadAffinity for Platform {
    /// A system with more than 64 cores needs another API on Windows that is
    /// not currently implemented.
    fn active_cpus() -> Vec<CoreId> {
        let mask = get_affinity_mask();
        (0..64_u64)
            .filter(|&i| (mask & 1 << i) == 1 << i)
            .map(|i| CoreId(i as usize))
            .collect()
    }

    fn physical_core(_cpu: CoreId) -> Option<CoreId> {
        // Needs `GetLogicalProcessorInformationEx`, not implemented yet.
        None
    }

    fn l1d_cache_size() -> Option<usize> {
        // Needs `GetLogicalProcessorInformation`, not implemented yet.
        None
    }

    fn pin_current(cpu: CoreId) {
        let mask: u64 = 1 << cpu.0;
        unsafe { SetThreadAffinityMask(GetCurrentThread(), mask as DWORD_PTR) };
    }
}

fn get_affinity_mask() -> u64 {
    let mut system_mask: usize = 0;
    let mut process_mask: usize = 0;

    unsafe {
        GetProcessAffinityMask(
            GetCurrentProcess(),
            &mut process_mask as PDWORD_PTR,
            &mut system_mask as PDWORD_PTR,
        );
    }

    process_mask as u64
}
