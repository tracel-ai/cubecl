use winapi::shared::basetsd::{DWORD_PTR, PDWORD_PTR};
use winapi::um::processthreadsapi::{GetCurrentProcess, GetCurrentThread};
use winapi::um::winbase::{GetProcessAffinityMask, SetThreadAffinityMask};

use super::CoreId;

/// System with more than 64 cores needs another API on windows that is not currently implemented
pub fn get_active_cores() -> impl Iterator<Item = CoreId> {
    let mask = get_affinity_mask();

    (0..64u64)
        .filter(move |&i| (mask & 1 << i) == 1 << i)
        .map(|i| CoreId(i as usize))
}

pub fn set_for_current(core_id: CoreId) {
    let mask: u64 = 1 << core_id.0;
    unsafe { SetThreadAffinityMask(GetCurrentThread(), mask as DWORD_PTR) };
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
