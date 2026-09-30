mod card_counters;

#[cfg(std_io)]
mod opened;
#[cfg(std_io)]
mod source;

#[cfg(all(std_io, target_os = "linux"))]
mod amdgpu_busy_percent;
#[cfg(all(std_io, target_os = "linux"))]
mod intel_idle_residency;
#[cfg(all(std_io, target_os = "linux"))]
mod nvml;

#[cfg(all(std_io, target_os = "windows"))]
mod gpu_engine_counters;
#[cfg(any(all(std_io, target_os = "windows"), test))]
mod gpu_engine_instance;

#[cfg(all(std_io, target_os = "macos"))]
mod io_accelerator;

pub use card_counters::CardCounters;
pub use cubecl_runtime::utilization::*;
