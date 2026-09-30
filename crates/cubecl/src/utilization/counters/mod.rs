mod opened;
mod registry;
mod source;

#[cfg(target_os = "linux")]
mod amdgpu_busy_percent;
#[cfg(target_os = "linux")]
mod intel_idle_residency;
#[cfg(target_os = "linux")]
mod nvml;

#[cfg(windows)]
mod gpu_engine_counters;
#[cfg(windows)]
mod gpu_engine_instance;

#[cfg(target_os = "macos")]
mod io_accelerator;

#[cfg(feature = "cpu")]
mod processor_times;

pub use registry::OpenedCounters;
