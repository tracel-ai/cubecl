use alloc::string::{String, ToString};
use core::fmt::{Display, Formatter};
use cubecl_ir::DeviceProperties;

/// The part a device is: what is measured on one device holds for every device of its part.
///
/// Where the card sits is left out on purpose. Identical cards differ in their index and their
/// [`physical`](cubecl_ir::DeviceIdentity::physical) address alone, and are one part; the index is
/// not even stable, since `CUDA_VISIBLE_DEVICES` makes whichever card was pinned index 0.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DevicePart {
    runtime: String,
    name: String,
    fingerprint: String,
    max_page_size: u64,
    parallelism: u32,
}

impl DevicePart {
    /// The part of a device of `runtime` reporting `properties`: its name and compile target, the
    /// largest allocation it takes and how many cores or multiprocessors it runs on, which tell
    /// apart parts that share a name.
    pub fn new(runtime: &str, properties: &DeviceProperties) -> Self {
        let hardware = &properties.hardware;
        Self {
            runtime: runtime.to_string(),
            name: properties.identity.name.clone(),
            fingerprint: properties.identity.fingerprint.clone(),
            max_page_size: properties.memory.max_page_size,
            parallelism: hardware
                .num_cpu_cores
                .or(hardware.num_streaming_multiprocessors)
                .unwrap_or(0),
        }
    }
}

/// One segment of a path: a runtime names itself `wgpu<spirv>`.
impl Display for DevicePart {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let segment = |text: &str| text.replace(|c: char| !c.is_ascii_alphanumeric(), "-");
        write!(
            f,
            "{}_{}_{}_mem{}_par{}",
            segment(&self.runtime),
            segment(&self.fingerprint),
            segment(&self.name),
            self.max_page_size,
            self.parallelism
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use cubecl_common::profile::TimingMethod;
    use cubecl_ir::{
        DeviceIdentity, HardwareProperties, MemoryDeviceProperties, PciAddress, PhysicalDevice,
        VectorSize, features::Features,
    };

    #[test]
    fn every_field_tells_parts_apart() {
        let part = rtx_2060();
        let others = vec![
            DevicePart {
                runtime: "wgpu<spirv>".to_string(),
                ..rtx_2060()
            },
            DevicePart {
                name: "NVIDIA GeForce GTX 1660 SUPER".to_string(),
                ..rtx_2060()
            },
            DevicePart {
                fingerprint: "ptx_sm86".to_string(),
                ..rtx_2060()
            },
            DevicePart {
                max_page_size: 12 << 30,
                ..rtx_2060()
            },
            DevicePart {
                parallelism: 4,
                ..rtx_2060()
            },
        ];

        for other in others {
            assert_ne!(part.to_string(), other.to_string(), "{other:?}");
        }
    }

    #[test]
    fn a_part_is_one_path_segment() {
        let part = DevicePart {
            runtime: "wgpu<spirv>".to_string(),
            name: "Intel(R) Arc(tm) B390 (PTL)".to_string(),
            fingerprint: "spirv_32902_45184".to_string(),
            ..rtx_2060()
        }
        .to_string();

        assert!(
            part.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "{part}"
        );
    }

    #[test]
    fn identical_cards_in_two_slots_are_one_part() {
        let in_slot = |bus| {
            let mut physical = PhysicalDevice::default();
            physical.pci_address = Some(PciAddress {
                domain: 0,
                bus,
                device: 0,
                function: 0,
            });
            let mut properties = l4();
            properties.identity.physical = Some(physical);
            DevicePart::new("cuda", &properties)
        };

        assert_eq!(in_slot(3), in_slot(4));
    }

    fn l4() -> DeviceProperties {
        let hardware = HardwareProperties {
            load_width: 128,
            vector_register_count: None,
            plane_size_min: 32,
            plane_size_max: 32,
            max_bindings: 32,
            max_shared_memory_size: 101376,
            max_cube_count: (u32::MAX, u16::MAX as u32, u16::MAX as u32),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 64),
            num_streaming_multiprocessors: Some(58),
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            last_level_cache_size: None,
            max_vector_size: VectorSize::MAX,
            cube_mma_reserved_shared_memory: 0,
        };
        DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties::new(5 << 30, 256),
            hardware,
            TimingMethod::Device,
            DeviceIdentity {
                name: "NVIDIA L4".to_string(),
                fingerprint: "ptx_sm89".to_string(),
                physical: None,
            },
        )
    }

    fn rtx_2060() -> DevicePart {
        DevicePart {
            runtime: "cuda".to_string(),
            name: "NVIDIA GeForce RTX 2060".to_string(),
            fingerprint: "ptx_sm75".to_string(),
            max_page_size: 6 << 30,
            parallelism: 30,
        }
    }
}
