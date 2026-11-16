//! Alone in its binary: the configuration is loaded once per process, and every other test
//! expects the device transport its runtime gives.
#![cfg(std_io)]

use cubecl_server::server::DeviceTransport;

#[test]
fn the_environment_turns_the_device_transport_off_without_loading_it() {
    // SAFETY: the only test in this binary, so no other thread reads the environment.
    unsafe { std::env::set_var("CUBECL_DEVICE_TRANSPORT", "off") };

    assert!(DeviceTransport::configured(|| panic!("the transport was loaded")).is_none());
}
