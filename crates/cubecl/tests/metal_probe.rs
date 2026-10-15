//! TEMPORARY, not for merging: what each device reports on a Mac.
//!
//! ```text
//! cargo test -p cubecl --features metal-native,wgpu --test metal_probe -- --ignored --nocapture
//! ioreg -r -c IOAccelerator -d 1 | grep PerformanceStatistics
//! ```

use cubecl::Device;
use cubecl::device::MetalDevice;

#[test]
#[ignore = "prints this machine's devices"]
fn print_devices() {
    let print = |device: &Device| {
        println!(
            "{:?}\n  name        {:?}\n  utilization {:?}\n  temperature {:?}\n  power       {:?}",
            device.runtime(),
            device.name(),
            device.utilization(),
            device.temperature(),
            device.power(),
        );
    };
    for device in Device::enumerate_all() {
        print(&device);
    }
    // Metal enumerates only its default device; each class is asked for by index.
    for kind in [MetalDevice::IntegratedGpu(0), MetalDevice::DiscreteGpu(0)] {
        match Device::metal(kind.clone()) {
            Ok(device) => print(&device),
            Err(missing) => println!("{kind:?}: not on this machine ({missing:?})"),
        }
    }
}
