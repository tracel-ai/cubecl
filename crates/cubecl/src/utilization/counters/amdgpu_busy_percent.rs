use alloc::format;
use alloc::string::ToString;
use std::path::PathBuf;

use cubecl_ir::PciAddress;

use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// Found through the card's PCI address, never by walking `card0`, `card1`, …: a DRM card's index
/// says nothing about which device the runtime drives.
#[derive(Debug)]
pub struct AmdgpuBusyPercentFile {
    path: PathBuf,
}

impl AmdgpuBusyPercentFile {
    pub fn new(pci_address: PciAddress) -> Self {
        Self {
            path: PathBuf::from(format!(
                "/sys/bus/pci/devices/{pci_address}/gpu_busy_percent"
            )),
        }
    }

    pub fn read(&self) -> DeviceUtilization {
        let reading = std::fs::read_to_string(&self.path)
            .map_err(|error| error.to_string())
            .and_then(|text| {
                text.trim()
                    .parse::<u32>()
                    .map_err(|error| error.to_string())
            });
        match reading {
            Ok(busy_percent) => DeviceUtilization::Measured { busy_percent },
            Err(message) => DeviceUtilization::Unavailable(UtilizationUnavailable::QueryFailed(
                format!("{}: {message}", self.path.display()),
            )),
        }
    }
}
