/// Configuration for the transport a runtime moves data between its devices with.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TransportConfig {
    /// Whether NCCL or RCCL carries the transfers between devices, `auto` unless configured.
    #[serde(default)]
    pub device: DeviceTransportMode,
}

/// Whether a runtime loads its device transport.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DeviceTransportMode {
    /// NCCL on CUDA or RCCL on HIP, where it is installed.
    #[default]
    #[serde(rename = "auto")]
    Auto,
    /// None: transfers between devices go through the host, and no collective group forms.
    #[serde(rename = "off")]
    Off,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CubeClRuntimeConfig;

    #[cfg(feature = "std")]
    #[test]
    fn a_config_file_without_the_section_loads_the_device_transport() {
        let config: CubeClRuntimeConfig = toml::from_str("").unwrap();

        assert_eq!(config.transport.device, DeviceTransportMode::Auto);
    }

    #[cfg(feature = "std")]
    #[test]
    fn the_device_transport_can_be_turned_off() {
        let config: CubeClRuntimeConfig = toml::from_str("[transport]\ndevice = \"off\"").unwrap();

        assert_eq!(config.transport.device, DeviceTransportMode::Off);
    }
}
