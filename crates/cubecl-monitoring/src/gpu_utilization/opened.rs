#[cfg(amdgpu_busy_percent)]
use super::amdgpu_sysfs::AmdgpuSysfsFiles;
#[cfg(gpu_engine_counters)]
use super::gpu_engine_counters::GpuEngineCounters;
#[cfg(intel_idle_residency)]
use super::intel_idle_residency::IntelIdleResidencyFiles;
#[cfg(io_accelerator)]
use super::io_accelerator::IoAcceleratorStatistics;
#[cfg(nvml)]
use super::nvml::NvmlUtilizationCounter;
use super::source::UtilizationSource;
use crate::utilization::{
    DevicePower, DeviceTemperature, DeviceUtilization, UtilizationUnavailable,
};

/// A device's counter once opened: a counter that would not open keeps the reason, so a device is
/// only ever opened once.
#[derive(Debug)]
pub enum OpenedCounter {
    #[cfg(nvml)]
    Nvml(NvmlUtilizationCounter),
    #[cfg(amdgpu_busy_percent)]
    AmdgpuSysfsFiles(AmdgpuSysfsFiles),
    #[cfg(intel_idle_residency)]
    IntelIdleResidencyFiles(IntelIdleResidencyFiles),
    #[cfg(gpu_engine_counters)]
    GpuEngineCounters(GpuEngineCounters),
    #[cfg(io_accelerator)]
    IoAcceleratorStatistics(IoAcceleratorStatistics),
    Unavailable(UtilizationUnavailable),
}

impl OpenedCounter {
    pub fn open(source: UtilizationSource) -> Self {
        let opened = match source {
            #[cfg(nvml)]
            UtilizationSource::Nvml(pci_address) => {
                NvmlUtilizationCounter::new(pci_address).map(Self::Nvml)
            }
            #[cfg(amdgpu_busy_percent)]
            UtilizationSource::AmdgpuSysfsFiles(pci_address) => {
                Ok(Self::AmdgpuSysfsFiles(AmdgpuSysfsFiles::new(pci_address)))
            }
            #[cfg(intel_idle_residency)]
            UtilizationSource::IntelIdleResidencyFiles(pci_address) => {
                IntelIdleResidencyFiles::new(pci_address).map(Self::IntelIdleResidencyFiles)
            }
            #[cfg(gpu_engine_counters)]
            UtilizationSource::GpuEngineCounters(luid) => {
                GpuEngineCounters::new(luid).map(Self::GpuEngineCounters)
            }
            #[cfg(io_accelerator)]
            UtilizationSource::IoAcceleratorStatistics(registry_entry_id) => {
                IoAcceleratorStatistics::new(registry_entry_id).map(Self::IoAcceleratorStatistics)
            }
            UtilizationSource::Unavailable(reason) => Err(reason),
        };
        opened.unwrap_or_else(Self::Unavailable)
    }

    pub fn read(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        match self {
            #[cfg(nvml)]
            Self::Nvml(counter) => counter.read(),
            #[cfg(amdgpu_busy_percent)]
            Self::AmdgpuSysfsFiles(file) => file.utilization(),
            #[cfg(intel_idle_residency)]
            Self::IntelIdleResidencyFiles(files) => files.read(),
            #[cfg(gpu_engine_counters)]
            Self::GpuEngineCounters(counters) => counters.read(),
            #[cfg(io_accelerator)]
            Self::IoAcceleratorStatistics(statistics) => statistics.utilization(),
            Self::Unavailable(reason) => Err(reason.clone()),
        }
    }

    /// The card's temperature, from the counters that keep one.
    pub fn temperature(&self) -> Result<DeviceTemperature, UtilizationUnavailable> {
        match self {
            #[cfg(nvml)]
            Self::Nvml(counter) => counter.temperature(),
            #[cfg(amdgpu_busy_percent)]
            Self::AmdgpuSysfsFiles(file) => file.temperature(),
            #[cfg(io_accelerator)]
            Self::IoAcceleratorStatistics(statistics) => statistics.temperature(),
            Self::Unavailable(reason) => Err(reason.clone()),
            #[allow(unreachable_patterns)]
            _ => Err(UtilizationUnavailable::NoSuchReading),
        }
    }

    /// The card's power draw, from the counters that keep one.
    pub fn power(&self) -> Result<DevicePower, UtilizationUnavailable> {
        match self {
            #[cfg(nvml)]
            Self::Nvml(counter) => counter.power(),
            #[cfg(amdgpu_busy_percent)]
            Self::AmdgpuSysfsFiles(file) => file.power(),
            #[cfg(io_accelerator)]
            Self::IoAcceleratorStatistics(statistics) => statistics.power(),
            Self::Unavailable(reason) => Err(reason.clone()),
            #[allow(unreachable_patterns)]
            _ => Err(UtilizationUnavailable::NoSuchReading),
        }
    }
}
