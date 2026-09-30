use alloc::boxed::Box;
use std::sync::{Mutex, PoisonError};

use sysinfo::System;

use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// The time every core of the machine spent running, which the CPU runtime's device is.
#[derive(Debug)]
pub struct ProcessorTimes {
    readings: Mutex<ProcessorReadings>,
}

#[derive(Debug)]
struct ProcessorReadings {
    system: Box<System>,
    has_read_once: bool,
}

impl ProcessorTimes {
    pub fn new() -> Self {
        Self {
            readings: Mutex::new(ProcessorReadings {
                system: Box::new(System::new()),
                has_read_once: false,
            }),
        }
    }

    pub fn read(&self) -> DeviceUtilization {
        let mut readings = self.readings.lock().unwrap_or_else(PoisonError::into_inner);
        readings.system.refresh_cpu_usage();
        if !core::mem::replace(&mut readings.has_read_once, true) {
            return DeviceUtilization::Unavailable(UtilizationUnavailable::NoPreviousReading);
        }
        DeviceUtilization::Measured {
            busy_percent: readings.system.global_cpu_usage().clamp(0.0, 100.0).round() as u32,
        }
    }
}
