use crate::memory_management::{MemoryDeviceProperties, TopologyProperties};
use alloc::collections::BTreeSet;

/// Properties of what the device can do, like what [features](Feature) are
/// supported by it and what its memory properties are.
#[derive(Debug)]
pub struct DeviceProperties<Feature: Ord + Copy> {
    set: alloc::collections::BTreeSet<Feature>,
    memory: MemoryDeviceProperties,
    topology: TopologyProperties,
}

impl<Feature: Ord + Copy> DeviceProperties<Feature> {
    /// Create a new feature set with the given features and memory properties.
    pub fn new(
        features: &[Feature],
        memory_props: MemoryDeviceProperties,
        topology: TopologyProperties,
    ) -> Self {
        let mut set = BTreeSet::new();
        for feature in features {
            set.insert(*feature);
        }

        DeviceProperties {
            set,
            memory: memory_props,
            topology,
        }
    }

    /// Check if the provided [feature](Feature) is supported by the runtime.
    pub fn feature_enabled(&self, feature: Feature) -> bool {
        self.set.contains(&feature)
    }

    /// Register a [feature](Feature) supported by the compute server.
    ///
    /// This should only be used by a [runtime](Runtime) when initializing a device.
    pub fn register_feature(&mut self, feature: Feature) -> bool {
        self.set.insert(feature)
    }

    /// The memory properties of this client.
    pub fn memory_properties(&self) -> &MemoryDeviceProperties {
        &self.memory
    }

    /// The topology properties of this client.
    pub fn topology_properties(&self) -> &TopologyProperties {
        &self.topology
    }
}
