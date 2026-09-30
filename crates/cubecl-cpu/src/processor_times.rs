use std::sync::{Mutex, PoisonError};

use cubecl_server::utilization::{DeviceUtilization, UtilizationUnavailable};
use sysinfo::System;

/// The time every core of the machine spent running, which the CPU runtime's device is. The
/// machine has one set of cores, so the process keeps one reading of them.
pub struct ProcessorTimes;

static PREVIOUS_READING: Mutex<Option<System>> = Mutex::new(None);

impl ProcessorTimes {
    /// The share of the time since the previous reading the cores spent running, averaged over
    /// every core.
    pub fn read_machine_wide() -> Result<DeviceUtilization, UtilizationUnavailable> {
        let mut previous_reading = PREVIOUS_READING
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(system) = previous_reading.as_mut() else {
            let mut system = System::new();
            system.refresh_cpu_usage();
            *previous_reading = Some(system);
            return Err(UtilizationUnavailable::NoPreviousReading);
        };
        system.refresh_cpu_usage();
        Ok(DeviceUtilization::new(
            system.global_cpu_usage().clamp(0.0, 100.0).round() as u32,
        ))
    }
}
