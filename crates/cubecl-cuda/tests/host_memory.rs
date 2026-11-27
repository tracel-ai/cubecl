use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_server::runtime::Runtime;

#[test]
fn devices_read_one_import() {
    let clients: Vec<_> = (0..CudaRuntime::enumerate_all_devices().len())
        .map(|index| CudaRuntime::client(&CudaDevice::new(index)))
        .collect();
    cubecl_core::runtime_tests::host_memory::test_devices_read_one_import(&clients);
}
