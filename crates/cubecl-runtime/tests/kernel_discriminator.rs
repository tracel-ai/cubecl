#[allow(dead_code)]
mod dummy;

use cubecl::prelude::*;
use cubecl_core as cubecl;
use dummy::{DummyDevice, test_client};

#[cube(launch)]
fn comptime_kernel(#[comptime] factor: u32) {
    let _ = factor;
}

use comptime_kernel::ComptimeKernel;

#[test]
fn define_disambiguates_comptime_variants() {
    let client = test_client(&DummyDevice);
    let settings = KernelSettings::new(
        *CubeDim::new_single(),
        ExecutionMode::Checked,
        AddressType::U32,
    );

    let k1 = ComptimeKernel::new(
        settings.clone(),
        client.properties_shared(),
        client.target_properties_shared(),
        1,
    );
    let k2 = ComptimeKernel::new(
        settings,
        client.properties_shared(),
        client.target_properties_shared(),
        2,
    );

    let def1 = k1.define();
    let def2 = k2.define();

    assert_ne!(
        def1.settings.kernel_name, def2.settings.kernel_name,
        "Kernels with distinct comptime arguments must have distinct kernel names"
    );
    assert!(def1.settings.kernel_name.starts_with("comptime_kernel_"));
    assert!(def2.settings.kernel_name.starts_with("comptime_kernel_"));
}
