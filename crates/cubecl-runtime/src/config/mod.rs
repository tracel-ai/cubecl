/// The `[autotune]` section of `cubecl.toml`.
pub mod autotune;
/// The `[compilation]` section of `cubecl.toml`.
pub mod compilation;
/// The `[environment]` section of `cubecl.toml`.
pub mod environment;
/// The `[memory]` section of `cubecl.toml`.
pub mod memory;
/// The `[streaming]` section of `cubecl.toml`.
pub mod streaming;
/// The `[throughput]` section of `cubecl.toml`.
pub mod throughput;
/// The `[transport]` section of `cubecl.toml`.
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
