use cubecl_core::{
    compute::DebugInformation,
    prelude::*,
    server::{Binding, Handle},
    Feature, KernelId, MemoryConfiguration, WgpuCompilationOptions,
};
use cubecl_runtime::{
    debug::{DebugLogger, ProfileLevel},
    memory_management::MemoryDeviceProperties,
    server::{self, ComputeServer},
    storage::BindingResource,
    TimestampsError, TimestampsResult,
};

use hashbrown::HashMap;

#[derive(Debug)]
pub struct WgpuServer {
    pipelines: HashMap<KernelId, Arc<ComputePipeline>>,
    logger: DebugLogger,
    duration_profiled: Option<Duration>,
    stream: CraneLiftStream,
}
