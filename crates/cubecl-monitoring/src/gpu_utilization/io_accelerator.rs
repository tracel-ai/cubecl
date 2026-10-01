use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use cubecl_ir::RegistryEntryId;
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_io_kit::{
    IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperty, IORegistryEntryIDMatching,
    IOServiceGetMatchingService, IOServiceGetMatchingServices, IOServiceMatching, io_iterator_t,
    io_service_t,
};

use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// `kIOMainPortDefault` is null, and naming the symbol would tie the binary to macOS 12.
const DEFAULT_MAIN_PORT: u32 = 0;

/// Older AMD drivers name the device's utilization differently.
const UTILIZATION_KEYS: [&str; 2] = ["Device Utilization %", "GPU Activity(%)"];

/// The statistics a GPU's driver keeps on its `IOAccelerator` entry in the I/O Registry, which an
/// unprivileged process may read.
#[derive(Debug)]
pub struct IoAcceleratorStatistics {
    accelerator: io_service_t,
}

impl IoAcceleratorStatistics {
    pub fn new(registry_entry_id: Option<RegistryEntryId>) -> Result<Self, UtilizationUnavailable> {
        let accelerator = match registry_entry_id {
            Some(registry_entry_id) => Self::accelerator_with_entry_id(registry_entry_id)?,
            None => Self::the_only_accelerator()?,
        };
        Ok(Self { accelerator })
    }

    pub fn read(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        let performance_statistics = CFString::from_static_str("PerformanceStatistics");
        // SAFETY: the entry is retained by `self`, and the property is returned retained.
        let statistics = unsafe {
            IORegistryEntryCreateCFProperty(
                self.accelerator,
                Some(&performance_statistics),
                None,
                0,
            )
        };
        let Some(statistics) = statistics.and_then(|value| value.downcast::<CFDictionary>().ok())
        else {
            return Err(Self::query_failed(
                "the accelerator keeps no PerformanceStatistics",
            ));
        };
        // SAFETY: the keys of an I/O Registry property dictionary are strings.
        let statistics = unsafe { statistics.cast_unchecked::<CFString, CFType>() };
        let busy_percent = UTILIZATION_KEYS.iter().find_map(|&key| {
            statistics
                .get(&CFString::from_static_str(key))?
                .downcast_ref::<CFNumber>()?
                .as_f64()
        });
        busy_percent
            .map(|busy_percent| DeviceUtilization::new(busy_percent as f32))
            .ok_or_else(|| {
                Self::query_failed("the PerformanceStatistics hold no device utilization")
            })
    }

    fn accelerator_with_entry_id(
        registry_entry_id: RegistryEntryId,
    ) -> Result<io_service_t, UtilizationUnavailable> {
        // SAFETY: an I/O Registry matching dictionary is a `CFDictionary`, which the lookup
        // consumes.
        let accelerator = unsafe {
            let matching = IORegistryEntryIDMatching(registry_entry_id.get())
                .map(|matching| CFRetained::cast_unchecked::<CFDictionary>(matching));
            IOServiceGetMatchingService(DEFAULT_MAIN_PORT, matching)
        };
        match accelerator {
            0 => Err(UtilizationUnavailable::QueryFailed(format!(
                "no I/O Registry entry has the id {:#x}",
                registry_entry_id.get()
            ))),
            accelerator => Ok(accelerator),
        }
    }

    /// Where the runtime reports no registry entry id, the accelerator can only be told apart
    /// from the others by being the only one, as on every Apple silicon Mac.
    fn the_only_accelerator() -> Result<io_service_t, UtilizationUnavailable> {
        let mut accelerators = Self::every_accelerator()?;
        match accelerators.len() {
            1 => Ok(accelerators.remove(0)),
            0 => Err(UtilizationUnavailable::QueryFailed(
                "the I/O Registry holds no IOAccelerator".to_string(),
            )),
            _ => {
                for accelerator in accelerators {
                    IOObjectRelease(accelerator);
                }
                Err(UtilizationUnavailable::CardAddressNotReported)
            }
        }
    }

    fn every_accelerator() -> Result<Vec<io_service_t>, UtilizationUnavailable> {
        let mut iterator: io_iterator_t = 0;
        // SAFETY: the class name is nul-terminated, the matching dictionary is a `CFDictionary`
        // the lookup consumes, and the iterator lands in `iterator`.
        let status = unsafe {
            let matching = IOServiceMatching(c"IOAccelerator".as_ptr())
                .map(|matching| CFRetained::cast_unchecked::<CFDictionary>(matching));
            IOServiceGetMatchingServices(DEFAULT_MAIN_PORT, matching, &mut iterator)
        };
        if status != 0 {
            return Err(UtilizationUnavailable::QueryFailed(format!(
                "IOServiceGetMatchingServices returned {status:#x}"
            )));
        }
        let accelerators = core::iter::from_fn(|| match IOIteratorNext(iterator) {
            0 => None,
            accelerator => Some(accelerator),
        })
        .collect();
        IOObjectRelease(iterator);
        Ok(accelerators)
    }

    fn query_failed(message: &str) -> UtilizationUnavailable {
        UtilizationUnavailable::QueryFailed(message.to_string())
    }
}

impl Drop for IoAcceleratorStatistics {
    fn drop(&mut self) {
        IOObjectRelease(self.accelerator);
    }
}
