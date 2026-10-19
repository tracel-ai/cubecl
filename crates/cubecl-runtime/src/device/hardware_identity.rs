use alloc::format;
use alloc::string::String;
use core::fmt::{Display, Formatter};
use cubecl_ir::{DeviceIdentity, DeviceProperties};

/// Which hardware a device is: its [`DeviceIdentity`] without the card it sits in, so what is
/// measured on one device holds for every device of the same hardware.
///
/// Its slot is left out, since `CUDA_VISIBLE_DEVICES` renumbers cards, and so is its memory, which
/// a cgroup limit or ECC moves without changing which kernel wins.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HardwareIdentity {
    runtime: String,
    name: String,
    fingerprint: String,
    parallelism: u32,
}

impl HardwareIdentity {
    /// The hardware of a device of `runtime` reporting `properties`. Its core or multiprocessor
    /// count tells apart hardware that shares a name.
    pub fn new(runtime: &str, properties: &DeviceProperties) -> Self {
        // Every field named, so a field added to `DeviceIdentity` is placed here or left out on
        // purpose.
        let DeviceIdentity {
            name,
            fingerprint,
            physical: _,
        } = &properties.identity;
        let hardware = &properties.hardware;
        Self {
            runtime: segment(runtime),
            name: segment(name),
            fingerprint: segment(fingerprint),
            parallelism: hardware
                .num_cpu_cores
                .or(hardware.num_streaming_multiprocessors)
                .unwrap_or(0),
        }
    }

    /// This identity written with `memory` installed, laid out as the throughput cache names its tables:
    /// another layout would orphan every peak measured.
    pub(crate) fn name_with_memory(&self, memory: u64) -> String {
        format!(
            "{}_{}_{}_mem{memory}_par{}",
            self.runtime, self.fingerprint, self.name, self.parallelism
        )
    }
}

/// `text` made fit for one path segment, since a runtime names itself `wgpu<spirv>`. Done at
/// construction, so two identities are equal exactly when they name the same table.
fn segment(text: &str) -> String {
    text.replace(|c: char| !c.is_ascii_alphanumeric(), "-")
}

impl Display for HardwareIdentity {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}_{}_{}_par{}",
            self.runtime, self.fingerprint, self.name, self.parallelism
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use cubecl_common::profile::TimingMethod;
    use cubecl_ir::{HardwareProperties, MemoryDeviceProperties, VectorSize, features::Features};

    #[test]
    fn a_gpu_is_identified_by_its_runtime_target_name_and_multiprocessors() {
        assert_eq!(
            HardwareIdentity::new("cuda", &l4()),
            HardwareIdentity {
                runtime: "cuda".to_string(),
                name: "NVIDIA-L4".to_string(),
                fingerprint: "ptx-sm89".to_string(),
                parallelism: 58,
            }
        );
    }

    #[test]
    fn cores_count_ahead_of_multiprocessors() {
        let mut properties = l4();
        properties.hardware.num_cpu_cores = Some(8);

        assert_eq!(HardwareIdentity::new("cuda", &properties).parallelism, 8);
    }

    #[test]
    fn names_that_differ_only_in_punctuation_identify_the_same_hardware() {
        let named = |name: &str| {
            let mut properties = l4();
            properties.identity.name = name.to_string();
            HardwareIdentity::new("wgpu<spirv>", &properties)
        };

        assert_eq!(
            named("Intel(R) Arc(tm) A770"),
            named("Intel-R- Arc-tm- A770")
        );
    }

    #[test]
    fn an_identity_is_one_path_segment() {
        let mut properties = l4();
        properties.identity.name = "Intel(R) Arc(tm) B390 (PTL)".to_string();
        properties.identity.fingerprint = "spirv_32902_45184".to_string();
        let written = HardwareIdentity::new("wgpu<spirv>", &properties).to_string();

        assert!(
            written
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "{written}"
        );
    }

    /// Throughput tables are named this way: another layout probes every peak again.
    #[test]
    fn the_name_with_memory_keeps_the_throughput_layout() {
        assert_eq!(
            HardwareIdentity::new("cuda", &l4()).name_with_memory(5 << 30),
            "cuda_ptx-sm89_NVIDIA-L4_mem5368709120_par58"
        );
    }

    #[test]
    fn every_field_tells_hardware_apart() {
        let identity = rtx_2060();
        let others = vec![
            HardwareIdentity {
                runtime: "wgpu-spirv-".to_string(),
                ..rtx_2060()
            },
            HardwareIdentity {
                name: "NVIDIA-GeForce-GTX-1660-SUPER".to_string(),
                ..rtx_2060()
            },
            HardwareIdentity {
                fingerprint: "ptx-sm86".to_string(),
                ..rtx_2060()
            },
            HardwareIdentity {
                parallelism: 4,
                ..rtx_2060()
            },
        ];

        for other in others {
            assert_ne!(identity.to_string(), other.to_string(), "{other:?}");
        }
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

    fn rtx_2060() -> HardwareIdentity {
        HardwareIdentity {
            runtime: "cuda".to_string(),
            name: "NVIDIA-GeForce-RTX-2060".to_string(),
            fingerprint: "ptx-sm75".to_string(),
            parallelism: 30,
        }
    }
}
