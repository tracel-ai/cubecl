//! TEMPORARY: prints the name each device gives itself, to check
//! `Device::name` on hardware this branch was not built on. Not for merging.
//!
//! On a Mac:
//!
//! ```text
//! cargo test -p cubecl --features metal-native,wgpu,cpu --test device_names -- --ignored --nocapture
//! ```
//!
//! Elsewhere, name the runtimes the machine has, e.g. `--features cuda,wgpu,cpu`.

use cubecl::Device;

#[test]
#[ignore = "prints this machine's devices"]
fn print_device_names() {
    println!("every device of every linked runtime:");
    for device in Device::enumerate_all() {
        println!(
            "  {:?} {:?}: {:?}",
            device.runtime(),
            device.to_id(),
            device.name()
        );
    }

    // Metal enumerates only its default device; each class is named by index.
    #[cfg(all(feature = "metal-native", target_vendor = "apple"))]
    {
        use cubecl::device::MetalDevice;

        println!("native Metal, by class:");
        for kind in [
            MetalDevice::DefaultDevice,
            MetalDevice::IntegratedGpu(0),
            MetalDevice::DiscreteGpu(0),
        ] {
            match Device::metal(kind.clone()) {
                Ok(device) => println!("  {kind:?}: {:?}", device.name()),
                Err(missing) => println!("  {kind:?}: not on this machine ({missing:?})"),
            }
        }
    }
}
