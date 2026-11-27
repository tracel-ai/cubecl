#![cfg(any(target_os = "linux", target_os = "windows"))]

use cubecl_core::prelude::Client;
use cubecl_hip::AmdDevice;
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;

/// The shared test passes without reading anything on a device lacking the Feature.
#[test]
fn kernel_reads_imported_host_memory() {
    let client = HipRuntime::client(&Default::default());
    cubecl_core::runtime_tests::host_memory::test_kernel_reads_imported_ranges(client);
}

fn clients() -> Vec<Client> {
    (0..HipRuntime::enumerate_all_devices().len())
        .map(|index| HipRuntime::client(&AmdDevice::new(index)))
        .collect()
}

#[test]
fn devices_read_one_import() {
    cubecl_core::runtime_tests::host_memory::test_devices_read_one_import(&clients());
}

#[test]
fn an_import_another_device_reads_stays_pinned() {
    cubecl_core::runtime_tests::host_memory::test_an_import_another_device_reads_stays_pinned(
        &clients(),
    );
}

#[test]
fn an_import_every_device_let_go_of_pins_again() {
    cubecl_core::runtime_tests::host_memory::test_an_import_every_device_let_go_of_pins_again(
        &clients(),
    );
}
