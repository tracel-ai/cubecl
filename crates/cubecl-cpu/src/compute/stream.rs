use crate::compute::{
    schedule::ScheduleTask,
    scheduler::KernelRunner,
    server::{CpuContext, CpuServer},
};
use cubecl_core::{MemoryConfiguration, server::ServerUtilities};
use cubecl_runtime::{
    memory_management::{MemoryDeviceProperties, MemoryManagement, MemoryManagementOptions},
    storage::BytesStorage,
};
use std::sync::Arc;

pub struct CpuStream {
    ctx: CpuContext,
    runner: KernelRunner,
    utilities: Arc<ServerUtilities<CpuServer>>,
}

impl core::fmt::Debug for CpuStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuStream").finish()
    }
}

impl CpuStream {
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<CpuServer>>,
    ) -> Self {
        let memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            memory_config,
            utilities.logger.clone(),
            MemoryManagementOptions::new("Main CPU"),
        );
        let memory_management_shared_memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            MemoryConfiguration::ExclusivePages,
            utilities.logger.clone(),
            MemoryManagementOptions::new("Shared Memory"),
        );

        let ctx = CpuContext::new(memory_management, memory_management_shared_memory);

        Self {
            ctx,
            runner: KernelRunner::default(),
            utilities,
        }
    }

    pub fn enqueue_task(&mut self, task: ScheduleTask) {}
    pub fn flush(&mut self) {}
}
