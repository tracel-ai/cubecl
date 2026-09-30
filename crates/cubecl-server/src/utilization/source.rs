#[cfg(target_os = "windows")]
use cubecl_ir::AdapterLuid;
use cubecl_ir::PhysicalDevice;
#[cfg(target_os = "macos")]
use cubecl_ir::RegistryEntryId;
#[cfg(target_os = "linux")]
use cubecl_ir::{PciAddress, PciVendor};

use crate::utilization::UtilizationUnavailable;

/// The counter a device's utilization is read from. The card and the platform decide it, never
/// the runtime: an NVIDIA card driven through wgpu reads the same counter as through CUDA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UtilizationSource {
    #[cfg(target_os = "linux")]
    Nvml(PciAddress),
    #[cfg(target_os = "linux")]
    AmdgpuBusyPercentFile(PciAddress),
    #[cfg(target_os = "linux")]
    IntelIdleResidencyFiles(PciAddress),
    #[cfg(target_os = "windows")]
    GpuEngineCounters(AdapterLuid),
    /// Without a registry entry id, the machine's only accelerator.
    #[cfg(target_os = "macos")]
    IoAcceleratorStatistics(Option<RegistryEntryId>),
    Unavailable(UtilizationUnavailable),
}

impl UtilizationSource {
    pub fn of_card(card: Option<&PhysicalDevice>) -> Self {
        match card {
            Some(card) => Self::counter_this_platform_keeps_for(card),
            None => Self::Unavailable(UtilizationUnavailable::NoCard),
        }
    }

    #[cfg(target_os = "linux")]
    fn counter_this_platform_keeps_for(card: &PhysicalDevice) -> Self {
        let counter_at: fn(PciAddress) -> Self = match card.vendor {
            Some(PciVendor::Nvidia) => Self::Nvml,
            Some(PciVendor::Amd) => Self::AmdgpuBusyPercentFile,
            Some(PciVendor::Intel) => Self::IntelIdleResidencyFiles,
            vendor => return Self::Unavailable(UtilizationUnavailable::NoCounterForCard(vendor)),
        };
        match card.pci_address {
            Some(pci_address) => counter_at(pci_address),
            None => Self::Unavailable(UtilizationUnavailable::CardAddressNotReported),
        }
    }

    #[cfg(target_os = "windows")]
    fn counter_this_platform_keeps_for(card: &PhysicalDevice) -> Self {
        match card.luid {
            Some(luid) => Self::GpuEngineCounters(luid),
            None => Self::Unavailable(UtilizationUnavailable::CardAddressNotReported),
        }
    }

    #[cfg(target_os = "macos")]
    fn counter_this_platform_keeps_for(card: &PhysicalDevice) -> Self {
        Self::IoAcceleratorStatistics(card.registry_entry_id)
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    fn counter_this_platform_keeps_for(card: &PhysicalDevice) -> Self {
        Self::Unavailable(UtilizationUnavailable::NoCounterForCard(card.vendor))
    }
}
