//! Relocation on the device, see [`cubecl_core::runtime_tests::relocation`].
//! One test per binary: each reads what the whole memory holds.

use cubecl_core::runtime_tests::relocation;
use cubecl_hip::HipRuntime;

#[test]
fn relocation_moves_live_bytes_off_outdated_pages() {
    relocation::relocation_moves_live_bytes_off_outdated_pages::<HipRuntime>();
}
