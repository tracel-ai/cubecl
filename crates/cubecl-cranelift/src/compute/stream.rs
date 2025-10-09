use cubecl_core::MemoryConfiguration;
use cubecl_runtime::{
    memory_management::{
        self, MemoryDeviceProperties, MemoryHandle, MemoryManagement, MemoryPoolOptions,
        SliceBinding, SliceHandle,
    },
    TimestampsError, TimestampsResult,
};
use std::{future::Future, num::NonZeroU64, pin::Pin, sync::Arc, time::Duration};
use web_time::Instant;

#[derive(Debug)]
pub struct CraneliftStream {}
