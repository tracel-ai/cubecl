use super::{
    autotune::AutotuneConfig, compilation::CompilationConfig, environment::EnvironmentConfig,
    memory::MemoryConfig, streaming::StreamingConfig, throughput::ThroughputConfig,
    transport::TransportConfig,
};
use alloc::sync::Arc;
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;

/// Static mutex holding the global configuration, initialized as `None`.
static CUBE_GLOBAL_CONFIG: Mutex<Option<Arc<CubeClRuntimeConfig>>> = Mutex::new(None);

/// Every setting of `CubeCL`, as `cubecl.toml` holds them, one section per field.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CubeClRuntimeConfig {
    /// The `[autotune]` section.
    #[serde(default)]
    pub autotune: AutotuneConfig,

    /// The `[throughput]` section.
    #[serde(default)]
    pub throughput: ThroughputConfig,

    /// The `[compilation]` section.
    #[serde(default)]
    pub compilation: CompilationConfig,

    /// The `[streaming]` section.
    #[serde(default)]
    pub streaming: StreamingConfig,

    /// The `[memory]` section.
    #[serde(default)]
    pub memory: MemoryConfig,

    /// The `[environment]` section.
    #[serde(default)]
    pub environment: EnvironmentConfig,

    /// The `[transport]` section.
    #[serde(default)]
    pub transport: TransportConfig,
}

impl RuntimeConfig for CubeClRuntimeConfig {
    fn storage() -> &'static Mutex<Option<Arc<Self>>> {
        &CUBE_GLOBAL_CONFIG
    }

    fn on_loaded(&self) {
        cubecl_environment::stream::set_policy_from_config(self.streaming.policy);
        cubecl_environment::records::configure(self.environment.records);
        // Before any device is initialized, so every cache opened afterwards
        // lands in the chosen environment.
        cubecl_environment::environment::activate(&self.environment.name);
        #[cfg(std_io)]
        cubecl_environment::environment::set_root(self.environment.path.root());
    }

    fn file_names() -> &'static [&'static str] {
        &["cubecl.toml", "CubeCL.toml"]
    }

    fn section_file_names() -> &'static [(&'static str, &'static str)] {
        &[("burn.toml", "cubecl"), ("Burn.toml", "cubecl")]
    }

    #[cfg(std_io)]
    fn override_from_env(self) -> Self {
        super::env::override_from_env(self)
    }
}
