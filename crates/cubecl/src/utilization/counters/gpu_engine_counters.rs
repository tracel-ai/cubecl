use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ptr;
use std::sync::{Mutex, PoisonError};

use cubecl_ir::AdapterLuid;
use windows_sys::Win32::System::Performance::{
    PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE,
    PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PDH_NO_DATA, PdhAddEnglishCounterW, PdhCloseQuery,
    PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
};

use super::gpu_engine_instance::{GpuEngine, GpuEngineInstance};
use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

const GPU_ENGINE_UTILIZATION_COUNTER_PATH: &str = "\\GPU Engine(*)\\Utilization Percentage";

/// The counters Task Manager reads a GPU's percent from, which cover every vendor. Like Task
/// Manager, the adapter is as busy as its busiest engine, summed over the processes using it.
#[derive(Debug)]
pub struct GpuEngineCounters {
    adapter: AdapterLuid,
    query: Mutex<GpuEngineQuery>,
}

#[derive(Debug)]
struct GpuEngineQuery {
    query: PDH_HQUERY,
    counter: PDH_HCOUNTER,
    has_collected_once: bool,
    /// Eight-byte words, the alignment of the items PDH writes into it.
    item_buffer: Vec<u64>,
    instance_name: String,
    busy_percent_by_engine: BTreeMap<GpuEngine, f64>,
}

// SAFETY: PDH handles may be used from any thread; the mutex around the query keeps it to one at a
// time.
unsafe impl Send for GpuEngineQuery {}

impl GpuEngineCounters {
    pub fn new(adapter: AdapterLuid) -> Result<Self, UtilizationUnavailable> {
        let mut query = GpuEngineQuery {
            query: ptr::null_mut(),
            counter: ptr::null_mut(),
            has_collected_once: false,
            item_buffer: Vec::new(),
            instance_name: String::new(),
            busy_percent_by_engine: BTreeMap::new(),
        };
        // SAFETY: a null data source is the live system, and the handle lands in `query`, which
        // closes it on drop.
        let status = unsafe { PdhOpenQueryW(ptr::null(), 0, &mut query.query) };
        GpuEngineQuery::succeeded("PdhOpenQueryW", status)
            .map_err(UtilizationUnavailable::QueryFailed)?;
        let counter_path: Vec<u16> = GPU_ENGINE_UTILIZATION_COUNTER_PATH
            .encode_utf16()
            .chain([0])
            .collect();
        // SAFETY: the path is nul-terminated and outlives the call.
        let status = unsafe {
            PdhAddEnglishCounterW(query.query, counter_path.as_ptr(), 0, &mut query.counter)
        };
        GpuEngineQuery::succeeded("PdhAddEnglishCounterW", status)
            .map_err(UtilizationUnavailable::QueryFailed)?;
        Ok(Self {
            adapter,
            query: Mutex::new(query),
        })
    }

    pub fn read(&self) -> DeviceUtilization {
        self.query
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .busiest_engine_of(self.adapter)
            .unwrap_or_else(|message| {
                DeviceUtilization::Unavailable(UtilizationUnavailable::QueryFailed(message))
            })
    }
}

impl GpuEngineQuery {
    fn busiest_engine_of(&mut self, adapter: AdapterLuid) -> Result<DeviceUtilization, String> {
        // SAFETY: the query is open for as long as `self` lives.
        let status = unsafe { PdhCollectQueryData(self.query) };
        Self::succeeded("PdhCollectQueryData", status)?;
        // A utilization percentage is a rate, which one collection has nothing to measure against.
        if !core::mem::replace(&mut self.has_collected_once, true) {
            return Ok(DeviceUtilization::Unavailable(
                UtilizationUnavailable::NoPreviousReading,
            ));
        }
        let item_count = self.collect_formatted_items()?;
        self.sum_busy_percent_by_engine_of(adapter, item_count);
        let busiest_engine = self
            .busy_percent_by_engine
            .values()
            .copied()
            .fold(0.0, f64::max);
        Ok(DeviceUtilization::Measured {
            busy_percent: busiest_engine.clamp(0.0, 100.0).round() as u32,
        })
    }

    /// Fills the item buffer, growing it for as long as PDH asks: the processes using the GPU come
    /// and go between two calls.
    fn collect_formatted_items(&mut self) -> Result<usize, String> {
        loop {
            let mut buffer_bytes = (self.item_buffer.len() * size_of::<u64>()) as u32;
            let mut item_count = 0;
            // SAFETY: PDH writes at most `buffer_bytes` into the buffer, which is aligned for the
            // items it writes.
            let status = unsafe {
                PdhGetFormattedCounterArrayW(
                    self.counter,
                    PDH_FMT_DOUBLE,
                    &mut buffer_bytes,
                    &mut item_count,
                    self.item_buffer.as_mut_ptr().cast(),
                )
            };
            match status {
                0 => return Ok(item_count as usize),
                PDH_NO_DATA => return Ok(0),
                PDH_MORE_DATA => self
                    .item_buffer
                    .resize((buffer_bytes as usize).div_ceil(size_of::<u64>()), 0),
                status => {
                    return Err(format!(
                        "PdhGetFormattedCounterArrayW returned {status:#010x}"
                    ));
                }
            }
        }
    }

    fn sum_busy_percent_by_engine_of(&mut self, adapter: AdapterLuid, item_count: usize) {
        self.busy_percent_by_engine.clear();
        // SAFETY: PDH wrote `item_count` items at the start of the buffer.
        let items = unsafe {
            core::slice::from_raw_parts(
                self.item_buffer
                    .as_ptr()
                    .cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(),
                item_count,
            )
        };
        for item in items {
            let status = item.FmtValue.CStatus;
            if status != PDH_CSTATUS_VALID_DATA && status != PDH_CSTATUS_NEW_DATA {
                continue;
            }
            Self::decode_instance_name(item.szName, &mut self.instance_name);
            let Some(instance) = GpuEngineInstance::parse(&self.instance_name) else {
                continue;
            };
            if instance.adapter != adapter {
                continue;
            }
            // SAFETY: the value was asked for as `PDH_FMT_DOUBLE`.
            let busy_percent = unsafe { item.FmtValue.Anonymous.doubleValue };
            *self
                .busy_percent_by_engine
                .entry(instance.engine)
                .or_default() += busy_percent;
        }
    }

    fn decode_instance_name(wide_name: *const u16, into: &mut String) {
        into.clear();
        if wide_name.is_null() {
            return;
        }
        // SAFETY: PDH's instance names are nul-terminated, in the buffer the items were written to.
        let length = (0..)
            .take_while(|&index| unsafe { *wide_name.add(index) } != 0)
            .count();
        // SAFETY: the `length` units before the terminator were just read.
        let units = unsafe { core::slice::from_raw_parts(wide_name, length) };
        into.extend(
            char::decode_utf16(units.iter().copied())
                .map(|unit| unit.unwrap_or(char::REPLACEMENT_CHARACTER)),
        );
    }

    fn succeeded(function: &str, status: u32) -> Result<(), String> {
        match status {
            0 => Ok(()),
            status => Err(format!("{function} returned {status:#010x}")),
        }
    }
}

impl Drop for GpuEngineQuery {
    fn drop(&mut self) {
        if !self.query.is_null() {
            // SAFETY: the query was opened by `PdhOpenQueryW` and is closed once, here.
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}
