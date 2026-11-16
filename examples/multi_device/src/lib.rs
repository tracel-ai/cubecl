use cubecl::{Device, prelude::Client};

/// A client for every device of the runtime the build links, in ascending device order, as a
/// collective group needs them.
pub fn clients() -> Vec<Client> {
    Device::enumerate_all()
        .into_iter()
        .filter(|device| match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => true,
            #[cfg(feature = "hip")]
            Device::Hip(_) => true,
            _ => false,
        })
        .map(|device| device.client())
        .collect()
}
