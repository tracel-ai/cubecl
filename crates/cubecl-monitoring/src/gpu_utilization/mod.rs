mod card_counters;

#[cfg(card_counter)]
mod opened;
#[cfg(card_counter)]
mod source;

#[cfg(amdgpu_busy_percent)]
mod amdgpu_sysfs;
#[cfg(intel_idle_residency)]
mod intel_idle_residency;
#[cfg(nvml)]
mod nvml;

#[cfg(gpu_engine_counters)]
mod gpu_engine_counters;
#[cfg(any(gpu_engine_counters, test))]
mod gpu_engine_instance;

#[cfg(io_accelerator)]
mod io_accelerator;

pub use card_counters::CardCounters;
#[cfg(io_accelerator)]
pub use io_accelerator::gpu_core_count;
