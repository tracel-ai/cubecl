use alloc::string::{String, ToString};
use core::fmt::{Display, Formatter};
use cubecl_ir::DeviceProperties;

/// The part a device is, never its index: `CUDA_VISIBLE_DEVICES` makes whichever card was pinned
/// index 0, and identical cards differ in nothing else. What is measured on a device holds for
/// every device of its part, and two cards this cannot separate are one part.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DevicePart {
    runtime: String,
    name: String,
    fingerprint: String,
    memory: u64,
    parallelism: u32,
}

impl DevicePart {
    /// The part of a device of `runtime` with `properties`.
    pub fn new(runtime: &str, properties: &DeviceProperties) -> Self {
        let hardware = &properties.hardware;
        Self {
            runtime: runtime.to_string(),
            name: properties.identity.name.clone(),
            fingerprint: properties.identity.fingerprint.clone(),
            memory: properties.memory.max_page_size,
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
            self.memory,
            self.parallelism
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn part(
        runtime: &str,
        name: &str,
        fingerprint: &str,
        memory: u64,
        parallelism: u32,
    ) -> DevicePart {
        DevicePart {
            runtime: runtime.to_string(),
            name: name.to_string(),
            fingerprint: fingerprint.to_string(),
            memory,
            parallelism,
        }
    }

    /// Two cards in one machine were served each other's peaks: pinning either
    /// makes it index 0, and the index was all that told them apart.
    #[test]
    fn two_parts_of_one_architecture_are_two_parts() {
        assert_ne!(
            part("cuda", "NVIDIA GeForce RTX 2060", "ptx_sm75", 6 << 30, 30).to_string(),
            part(
                "cuda",
                "NVIDIA GeForce GTX 1660 SUPER",
                "ptx_sm75",
                6 << 30,
                22
            )
            .to_string(),
        );
    }

    #[test]
    fn a_machine_that_gains_memory_is_another_part() {
        let xeon = "Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz";

        assert_ne!(
            part("cpu", xeon, "cpu_x86_64", 32 << 30, 16).to_string(),
            part("cpu", xeon, "cpu_x86_64", 64 << 30, 16).to_string(),
        );
    }

    #[test]
    fn a_container_given_fewer_cores_is_another_part() {
        let xeon = "Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz";

        assert_ne!(
            part("cpu", xeon, "cpu_x86_64", 64 << 30, 16).to_string(),
            part("cpu", xeon, "cpu_x86_64", 64 << 30, 4).to_string(),
        );
    }

    #[test]
    fn a_part_is_one_path_segment() {
        let part = part(
            "wgpu<spirv>",
            "Intel(R) Arc(tm) B390 (PTL)",
            "spirv_32902_45184",
            1 << 31,
            0,
        )
        .to_string();

        assert!(
            part.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "{part}"
        );
    }
}
