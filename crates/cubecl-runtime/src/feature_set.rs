use crate::memory_management::{HardwareProperties, MemoryDeviceProperties};
use alloc::collections::BTreeSet;

/// Properties of what the device can do, like what `Feature` are
/// supported by it and what its memory properties are.
#[derive(Debug)]
pub struct DeviceProperties<Feature: Ord + Copy> {
    set: alloc::collections::BTreeSet<Feature>,
    memory: MemoryDeviceProperties,
    hardware: HardwareProperties,
    timing_mode: TimingMode,
}

#[derive(Debug, Clone, Copy)]
pub enum TimingMode {
    /// Using the device own measuting capability.
    ///
    /// Normally compatible with async.
    Device,
    /// Using [std::time::Instant] to measure kernel execution.
    ///
    /// When this version is activated, we must await on async profiling.
    System,
}

impl<Feature: Ord + Copy> DeviceProperties<Feature> {
    /// Create a new feature set with the given features and memory properties.
    pub fn new(
        features: &[Feature],
        memory_props: MemoryDeviceProperties,
        hardware: HardwareProperties,
        profiling_mode: TimingMode,
    ) -> Self {
        let mut set = BTreeSet::new();
        for feature in features {
            set.insert(*feature);
        }

        DeviceProperties {
            set,
            memory: memory_props,
            hardware,
            timing_mode: profiling_mode,
        }
    }

    pub fn tiling_mode(&self) -> TimingMode {
        self.timing_mode
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

    /// The memory properties of this client.
    pub fn memory_properties(&self) -> &MemoryDeviceProperties {
        &self.memory
    }

    /// The topology properties of this client.
    pub fn hardware_properties(&self) -> &HardwareProperties {
        &self.hardware
    }
}
