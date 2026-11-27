#![cfg(any(target_os = "linux", target_os = "windows"))]

use cubecl_hip::AmdDevice;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;

/// The shared test passes without reading anything on a device lacking the Feature.
#[test]
fn kernel_reads_imported_host_memory() {
    let client = HipRuntime::client(&Default::default());
    cubecl_core::runtime_tests::host_memory::test_kernel_reads_imported_ranges(client);
}

#[test]
fn devices_read_one_import() {
    let clients: Vec<_> = (0..HipRuntime::enumerate_all_devices().len())
        .map(|index| HipRuntime::client(&AmdDevice::new(index)))
        .collect();
    cubecl_core::runtime_tests::host_memory::test_devices_read_one_import(&clients);
}
