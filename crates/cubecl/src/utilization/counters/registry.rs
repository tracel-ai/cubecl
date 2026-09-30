use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use std::sync::{Mutex, MutexGuard, PoisonError};

use cubecl_core::device::DeviceId;

use super::opened::OpenedCounter;
use super::source::UtilizationSource;
use crate::{Device, utilization::DeviceUtilization};

static OPENED_COUNTERS_BY_DEVICE: Mutex<BTreeMap<DeviceId, Arc<OpenedCounter>>> =
    Mutex::new(BTreeMap::new());

/// Every device's counter, opened the first time the device is asked and kept for the life of
/// the process: opening is the costly half, loading NVML or creating the device's client.
pub struct OpenedCounters;

impl OpenedCounters {
    pub fn read_opening_on_first_use(device: &Device) -> DeviceUtilization {
        Self::opened_for(device).read()
    }

    fn opened_for(device: &Device) -> Arc<OpenedCounter> {
        let device_id = device.to_id();
        if let Some(opened) = Self::lock().get(&device_id) {
            return opened.clone();
        }
        // Opened with the lock released, so a device whose client is slow to create holds up no
        // other device. Two threads racing here both open one, and the first to insert wins.
        let opened = Arc::new(OpenedCounter::open(UtilizationSource::of_device(device)));
        Self::lock().entry(device_id).or_insert(opened).clone()
    }

    fn lock() -> MutexGuard<'static, BTreeMap<DeviceId, Arc<OpenedCounter>>> {
        OPENED_COUNTERS_BY_DEVICE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}
