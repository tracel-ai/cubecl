use alloc::boxed::Box;
use alloc::string::{String, ToString};

use cubecl_ir::PciAddress;
use nvml_wrapper::Nvml;
use nvml_wrapper::error::NvmlError;

use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// NVIDIA's management library, which loads at run time, so a build that reaches for it starts
/// on a machine with no NVIDIA driver.
#[derive(Debug)]
pub struct NvmlUtilizationCounter {
    nvml: Box<Nvml>,
    pci_bus_id: String,
}

impl NvmlUtilizationCounter {
    pub fn new(pci_address: PciAddress) -> Result<Self, UtilizationUnavailable> {
        let nvml = Box::new(Nvml::init().map_err(unavailability_from_nvml_error)?);
        let pci_bus_id = pci_address.to_string();
        nvml.device_by_pci_bus_id(pci_bus_id.as_str())
            .map_err(unavailability_from_nvml_error)?;
        Ok(Self { nvml, pci_bus_id })
    }

    pub fn read(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        // NVML's device handle borrows the library, so it cannot live beside it in this struct;
        // the card is looked up again by its bus id.
        self.nvml
            .device_by_pci_bus_id(self.pci_bus_id.as_str())
            .and_then(|device| device.utilization_rates())
            .map(|utilization| DeviceUtilization::new(utilization.gpu as f32))
            .map_err(unavailability_from_nvml_error)
    }
}

fn unavailability_from_nvml_error(error: NvmlError) -> UtilizationUnavailable {
    match error {
        NvmlError::LibloadingError(_) | NvmlError::LibraryNotFound => {
            UtilizationUnavailable::DriverLibraryNotFound(error.to_string())
        }
        _ => UtilizationUnavailable::QueryFailed(error.to_string()),
    }
}
