use std::collections::BTreeSet;

use crate::memory_management::dynamic::MemoryDeviceProperties;


/// The set of [features](Feature) supported by a [runtime](Runtime).
#[derive(Debug)]
pub struct FeatureSet<Feature: Ord + Copy> {
    set: alloc::collections::BTreeSet<Feature>,
    memory: MemoryDeviceProperties
}

impl<Feature: Ord + Copy> FeatureSet<Feature> {
    /// Create a new feature set with the given features and memory properties.
    pub fn new(features: &[Feature], mem_props: MemoryDeviceProperties) -> Self {
        let mut set = BTreeSet::new();
        for feature in features {
            set.insert(*feature);
        }

        FeatureSet {
            set,
            memory: mem_props
        }
    }

    /// Check if the provided [feature](Feature) is supported by the runtime.
    pub fn enabled(&self, feature: Feature) -> bool {
        self.set.contains(&feature)
    }

    /// The memory properties of this device.
    pub fn memory_properties(&self) -> &MemoryDeviceProperties {
        &self.memory
    }

    /// Register a [feature](Feature) supported by the compute server.
    ///
    /// This should only be used by a [runtime](Runtime) when initializing a device.
    pub fn register(&mut self, feature: Feature) -> bool {
        self.set.insert(feature)
    }
}
