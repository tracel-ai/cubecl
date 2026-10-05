//! Relocation on the host, see [`cubecl_core::runtime_tests::relocation`].
//!
//! The checks run in one test, in turn: each reads what the whole memory
//! holds, which a test running beside it would change.

use cubecl_core::runtime_tests::relocation;
use cubecl_cpu::CpuRuntime;

#[test]
fn relocation() {
    relocation::relocation_moves_live_bytes_off_outdated_pages::<CpuRuntime>();
    // A read-back on the CPU points into the memory it read.
    relocation::a_held_read_back_keeps_its_allocation_in_place::<CpuRuntime>();
}
