use cubecl_core::prelude::Client;
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_server::runtime::Runtime;

fn clients() -> Vec<Client> {
    (0..CudaRuntime::enumerate_all_devices().len())
        .map(|index| CudaRuntime::client(&CudaDevice::new(index)))
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
