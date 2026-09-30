#[cfg(target_os = "linux")]
use super::amdgpu_busy_percent::AmdgpuBusyPercentFile;
#[cfg(target_os = "windows")]
use super::gpu_engine_counters::GpuEngineCounters;
#[cfg(target_os = "linux")]
use super::intel_idle_residency::IntelIdleResidencyFiles;
#[cfg(target_os = "macos")]
use super::io_accelerator::IoAcceleratorStatistics;
#[cfg(target_os = "linux")]
use super::nvml::NvmlUtilizationCounter;
use super::source::UtilizationSource;
use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// A device's counter once opened: a counter that would not open keeps the reason, so a device is
/// only ever opened once.
#[derive(Debug)]
pub enum OpenedCounter {
    #[cfg(target_os = "linux")]
    Nvml(NvmlUtilizationCounter),
    #[cfg(target_os = "linux")]
    AmdgpuBusyPercentFile(AmdgpuBusyPercentFile),
    #[cfg(target_os = "linux")]
    IntelIdleResidencyFiles(IntelIdleResidencyFiles),
    #[cfg(target_os = "windows")]
    GpuEngineCounters(GpuEngineCounters),
    #[cfg(target_os = "macos")]
    IoAcceleratorStatistics(IoAcceleratorStatistics),
    Unavailable(UtilizationUnavailable),
}

impl OpenedCounter {
    pub fn open(source: UtilizationSource) -> Self {
        let opened = match source {
            #[cfg(target_os = "linux")]
            UtilizationSource::Nvml(pci_address) => {
                NvmlUtilizationCounter::new(pci_address).map(Self::Nvml)
            }
            #[cfg(target_os = "linux")]
            UtilizationSource::AmdgpuBusyPercentFile(pci_address) => Ok(
                Self::AmdgpuBusyPercentFile(AmdgpuBusyPercentFile::new(pci_address)),
            ),
            #[cfg(target_os = "linux")]
            UtilizationSource::IntelIdleResidencyFiles(pci_address) => {
                IntelIdleResidencyFiles::new(pci_address).map(Self::IntelIdleResidencyFiles)
            }
            #[cfg(target_os = "windows")]
            UtilizationSource::GpuEngineCounters(luid) => {
                GpuEngineCounters::new(luid).map(Self::GpuEngineCounters)
            }
            #[cfg(target_os = "macos")]
            UtilizationSource::IoAcceleratorStatistics(registry_entry_id) => {
                IoAcceleratorStatistics::new(registry_entry_id).map(Self::IoAcceleratorStatistics)
            }
            UtilizationSource::Unavailable(reason) => Err(reason),
        };
        opened.unwrap_or_else(Self::Unavailable)
    }

    pub fn read(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Nvml(counter) => counter.read(),
            #[cfg(target_os = "linux")]
            Self::AmdgpuBusyPercentFile(file) => file.read(),
            #[cfg(target_os = "linux")]
            Self::IntelIdleResidencyFiles(files) => files.read(),
            #[cfg(target_os = "windows")]
            Self::GpuEngineCounters(counters) => counters.read(),
            #[cfg(target_os = "macos")]
            Self::IoAcceleratorStatistics(statistics) => statistics.read(),
            Self::Unavailable(reason) => Err(reason.clone()),
        }
    }
}
