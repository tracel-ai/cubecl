/// How autotune benchmarks its candidates and where it reports.
pub mod autotune;
/// How kernels compile: bounds checks, debug data, dumps and the compile threads.
pub mod compilation;
/// Which named environment the process warms into.
pub mod environment;
/// Which pools persistent allocations go to, and the memory log.
pub mod memory;
/// What the profiling log reports.
pub mod profiling;
/// How many streams a device runs and how they are chosen.
pub mod streaming;
/// Whether throughput measurements are cached.
pub mod throughput;
/// How a runtime moves data between its devices.
pub mod transport;

mod base;
mod debug_info;
#[cfg(std_io)]
mod env;
mod logger;
mod type_name;

pub use base::CubeClRuntimeConfig;
pub use cubecl_environment::config::RuntimeConfig;
#[cfg(std_io)]
pub use cubecl_environment::persistence::CacheConfig;
pub use cubecl_environment::stream::StreamPolicy;
pub use type_name::TypeNameFormatLevel;

pub(crate) use cubecl_environment::config::logger::{LogLevel, LoggerConfig};
pub(crate) use debug_info::effective_debug_info;
pub(crate) use logger::Logger;
pub(crate) use type_name::short_type_name;
