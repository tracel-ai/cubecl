use crate::memory_management::{HardwareProperties, MemoryDeviceProperties};
use alloc::collections::BTreeSet;
use cubecl_common::profile::TimingMethod;

/// Properties of what the device can do, like what `Feature` are
/// supported by it and what its memory properties are.
#[derive(Debug)]
pub struct DeviceProperties<Feature: Ord + Copy> {
    set: alloc::collections::BTreeSet<Feature>,
    /// The memory properties of this client.
    pub memory: MemoryDeviceProperties,
    /// The topology properties of this client.
    pub hardware: HardwareProperties,
    /// The method used for profiling on the device.
    pub timing_method: TimingMethod,
}

impl<Feature: Ord + Copy> DeviceProperties<Feature> {
    /// Create a new feature set with the given features and memory properties.
    pub fn new(
        features: &[Feature],
        memory_props: MemoryDeviceProperties,
        hardware: HardwareProperties,
        timing_method: TimingMethod,
    ) -> Self {
        let mut set = BTreeSet::new();
        for feature in features {
            set.insert(*feature);
        }

        DeviceProperties {
            set,
            memory: memory_props,
            hardware,
            timing_method,
        }
    }

    /// Check if the provided `Feature` is supported by the runtime.
    pub fn feature_enabled(&self, feature: Feature) -> bool {
        self.set.contains(&feature)
    }

    /// Register a `Feature` supported by the compute server.
    ///
    /// This should only be used by a [runtime](cubecl_core::Runtime) when initializing a device.
    pub fn register_feature(&mut self, feature: Feature) -> bool {
        self.set.insert(feature)
    }

    /// Removes a `Feature` from the compute server.
    ///
    /// This should only be used by a [runtime](cubecl_core::Runtime) when initializing a device.
    pub fn remove_feature(&mut self, feature: Feature) {
        self.set.remove(&feature);
    }
}
