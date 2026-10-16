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
