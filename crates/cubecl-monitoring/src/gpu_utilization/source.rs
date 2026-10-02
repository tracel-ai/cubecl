#[cfg(gpu_engine_counters)]
use cubecl_ir::AdapterLuid;
use cubecl_ir::PhysicalDevice;
#[cfg(io_accelerator)]
use cubecl_ir::RegistryEntryId;
#[cfg(any(amdgpu_busy_percent, intel_idle_residency, nvml))]
use cubecl_ir::{PciAddress, PciVendor};

use crate::utilization::UtilizationUnavailable;

/// The counter a device's utilization is read from, among the ones this build compiles. The card
/// and the platform decide it, never the runtime: an NVIDIA card driven through wgpu reads the same
/// counter as through CUDA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UtilizationSource {
    #[cfg(nvml)]
    Nvml(PciAddress),
    #[cfg(amdgpu_busy_percent)]
    AmdgpuBusyPercentFile(PciAddress),
    #[cfg(intel_idle_residency)]
    IntelIdleResidencyFiles(PciAddress),
    #[cfg(gpu_engine_counters)]
    GpuEngineCounters(AdapterLuid),
    /// Without a registry entry id, the machine's only accelerator.
    #[cfg(io_accelerator)]
    IoAcceleratorStatistics(Option<RegistryEntryId>),
    Unavailable(UtilizationUnavailable),
}

impl UtilizationSource {
    pub fn of_card(card: Option<&PhysicalDevice>) -> Self {
        match card {
            Some(card) => Self::counter_this_build_keeps_for(card),
            None => Self::Unavailable(UtilizationUnavailable::NoCard),
        }
    }

    #[cfg(any(amdgpu_busy_percent, intel_idle_residency, nvml))]
    fn counter_this_build_keeps_for(card: &PhysicalDevice) -> Self {
        let counter_at: fn(PciAddress) -> Self = match card.vendor {
            #[cfg(nvml)]
            Some(PciVendor::Nvidia) => Self::Nvml,
            #[cfg(amdgpu_busy_percent)]
            Some(PciVendor::Amd) => Self::AmdgpuBusyPercentFile,
            #[cfg(intel_idle_residency)]
            Some(PciVendor::Intel) => Self::IntelIdleResidencyFiles,
            vendor => return Self::Unavailable(UtilizationUnavailable::NoCounterForCard(vendor)),
        };
        match card.pci_address {
            Some(pci_address) => counter_at(pci_address),
            None => Self::Unavailable(UtilizationUnavailable::CardAddressNotReported),
        }
    }

    #[cfg(gpu_engine_counters)]
    fn counter_this_build_keeps_for(card: &PhysicalDevice) -> Self {
        match card.luid {
            Some(luid) => Self::GpuEngineCounters(luid),
            None => Self::Unavailable(UtilizationUnavailable::CardAddressNotReported),
        }
    }

    #[cfg(io_accelerator)]
    fn counter_this_build_keeps_for(card: &PhysicalDevice) -> Self {
        Self::IoAcceleratorStatistics(card.registry_entry_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_with_no_card_behind_it_has_no_counter() {
        assert_eq!(
            UtilizationSource::of_card(None),
            UtilizationSource::Unavailable(UtilizationUnavailable::NoCard)
        );
    }

    #[cfg(all(amdgpu_busy_percent, intel_idle_residency, nvml))]
    mod linux {
        use super::*;

        const ADDRESS: PciAddress = PciAddress {
            domain: 0,
            bus: 7,
            device: 0,
            function: 0,
        };

        fn source_of(
            vendor: Option<PciVendor>,
            pci_address: Option<PciAddress>,
        ) -> UtilizationSource {
            let mut card = PhysicalDevice::default();
            card.vendor = vendor;
            card.pci_address = pci_address;
            UtilizationSource::of_card(Some(&card))
        }

        #[test]
        fn a_vendor_with_a_counter_reads_it_at_the_cards_address() {
            assert_eq!(
                source_of(Some(PciVendor::Nvidia), Some(ADDRESS)),
                UtilizationSource::Nvml(ADDRESS)
            );
            assert_eq!(
                source_of(Some(PciVendor::Amd), Some(ADDRESS)),
                UtilizationSource::AmdgpuBusyPercentFile(ADDRESS)
            );
            assert_eq!(
                source_of(Some(PciVendor::Intel), Some(ADDRESS)),
                UtilizationSource::IntelIdleResidencyFiles(ADDRESS)
            );
        }

        #[test]
        fn a_vendor_without_a_counter_is_named_even_with_an_address() {
            assert_eq!(
                source_of(Some(PciVendor::Arm), Some(ADDRESS)),
                UtilizationSource::Unavailable(UtilizationUnavailable::NoCounterForCard(Some(
                    PciVendor::Arm
                )))
            );
            assert_eq!(
                source_of(None, Some(ADDRESS)),
                UtilizationSource::Unavailable(UtilizationUnavailable::NoCounterForCard(None))
            );
        }

        #[test]
        fn a_card_with_no_address_cannot_be_found() {
            assert_eq!(
                source_of(Some(PciVendor::Nvidia), None),
                UtilizationSource::Unavailable(UtilizationUnavailable::CardAddressNotReported)
            );
        }
    }

    #[cfg(gpu_engine_counters)]
    mod windows {
        use cubecl_ir::PciVendor;

        use super::*;

        fn source_of(vendor: Option<PciVendor>, luid: Option<AdapterLuid>) -> UtilizationSource {
            let mut card = PhysicalDevice::default();
            card.vendor = vendor;
            card.luid = luid;
            UtilizationSource::of_card(Some(&card))
        }

        #[test]
        fn a_card_with_a_luid_reads_the_gpu_engine_counters_whatever_its_vendor() {
            let luid = AdapterLuid::from_parts(0x0001_1d8a, 0);
            for vendor in [
                Some(PciVendor::Nvidia),
                Some(PciVendor::Amd),
                Some(PciVendor::Intel),
                Some(PciVendor::Qualcomm),
                None,
            ] {
                assert_eq!(
                    source_of(vendor, Some(luid)),
                    UtilizationSource::GpuEngineCounters(luid),
                    "{vendor:?}"
                );
            }
        }

        #[test]
        fn a_card_with_no_luid_cannot_be_found() {
            assert_eq!(
                source_of(Some(PciVendor::Amd), None),
                UtilizationSource::Unavailable(UtilizationUnavailable::CardAddressNotReported)
            );
        }
    }

    #[cfg(io_accelerator)]
    mod macos {
        use super::*;

        fn source_of(registry_entry_id: Option<RegistryEntryId>) -> UtilizationSource {
            let mut card = PhysicalDevice::default();
            card.registry_entry_id = registry_entry_id;
            UtilizationSource::of_card(Some(&card))
        }

        #[test]
        fn a_card_with_a_registry_entry_id_reads_that_entry() {
            let registry_entry_id = RegistryEntryId::new(0x1_0000_04c8);
            assert_eq!(
                source_of(Some(registry_entry_id)),
                UtilizationSource::IoAcceleratorStatistics(Some(registry_entry_id))
            );
        }

        #[test]
        fn a_card_with_no_registry_entry_id_reads_the_only_accelerator() {
            assert_eq!(
                source_of(None),
                UtilizationSource::IoAcceleratorStatistics(None)
            );
        }
    }
}
