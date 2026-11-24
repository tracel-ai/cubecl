use core::num::NonZeroU16;

/// Configuration for the transport a runtime moves data between its devices with.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TransportConfig {
    /// Whether NCCL or RCCL carries the transfers between devices, `auto` unless configured.
    #[serde(default)]
    pub device: DeviceTransportMode,
    /// How the provider sets up a communicator, the `[transport.provider]` section.
    #[serde(default)]
    pub provider: TransportProviderConfig,
}

/// How the transport provider, NCCL or RCCL, sets up a communicator. Ignored when the device
/// transport is off.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TransportProviderConfig {
    /// The fewest channels a communicator runs a collective on.
    #[serde(default)]
    pub min_channels: MinChannels,
}

/// A floor on the channels a communicator runs a collective on: the provider may pick more, and
/// `NCCL_MIN_CTAS` overrides it. Over PCI Express the provider's topology search settles on one
/// channel, which leaves the link idle between a block's transfers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct MinChannels(pub NonZeroU16);

impl Default for MinChannels {
    fn default() -> Self {
        Self(NonZeroU16::new(16).unwrap())
    }
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

    #[cfg(feature = "std")]
    #[test]
    fn a_config_file_without_the_section_floors_channels_at_sixteen() {
        let config: CubeClRuntimeConfig = toml::from_str("").unwrap();

        assert_eq!(config.transport.provider.min_channels.0.get(), 16);
    }

    #[cfg(feature = "std")]
    #[test]
    fn the_channel_floor_can_be_set() {
        let config: CubeClRuntimeConfig =
            toml::from_str("[transport.provider]\nmin_channels = 4").unwrap();

        assert_eq!(config.transport.provider.min_channels.0.get(), 4);
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_floor_of_no_channels_is_refused() {
        let config =
            toml::from_str::<CubeClRuntimeConfig>("[transport.provider]\nmin_channels = 0");

        assert!(config.is_err());
    }
}
