use crate::config::memory::MemoryConfig;
use crate::config::streaming::StreamingConfig;

use super::{
    autotune::AutotuneConfig, compilation::CompilationConfig, profiling::ProfilingConfig,
    throughput::ThroughputConfig, transport::TransportConfig,
};
use alloc::sync::Arc;
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;

/// Static mutex holding the global configuration, initialized as `None`.
static CUBE_GLOBAL_CONFIG: Mutex<Option<Arc<CubeClRuntimeConfig>>> = Mutex::new(None);

/// Represents the global configuration for `CubeCL`, combining profiling, autotuning, and compilation settings.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CubeClRuntimeConfig {
    /// Configuration for profiling `CubeCL` operations.
    #[serde(default)]
    pub profiling: ProfilingConfig,

    /// Configuration for autotuning performance parameters.
    #[serde(default)]
    pub autotune: AutotuneConfig,

    /// Configuration for throughput settings.
    #[serde(default)]
    pub throughput: ThroughputConfig,

    /// Configuration for compilation settings.
    #[serde(default)]
    pub compilation: CompilationConfig,

    /// Configuration for streaming settings.
    #[serde(default)]
    pub streaming: StreamingConfig,

    /// Configuration for memory settings.
    #[serde(default)]
    pub memory: MemoryConfig,

    /// Which named environment to warm into.
    #[serde(default)]
    pub environment: super::environment::EnvironmentConfig,

    /// Configuration for the transport between devices.
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
    fn override_from_env(mut self) -> Self {
        use super::compilation::{CompilationLogLevel, F16Evaluation};
        use super::env::{env_bool, env_variant};
        use crate::config::{
            autotune::{AutotuneLevel, AutotuneLogLevel},
            profiling::ProfilingLogLevel,
        };

        if let Ok(val) = std::env::var("CUBECL_DEBUG_LOG") {
            self.compilation.logger.level = CompilationLogLevel::Full;
            self.profiling.logger.level = ProfilingLogLevel::Medium;
            self.autotune.logger.level = AutotuneLogLevel::Full;

            match val.as_str() {
                "stdout" => {
                    self.compilation.logger.stdout = true;
                    self.profiling.logger.stdout = true;
                    self.autotune.logger.stdout = true;
                }
                "stderr" => {
                    self.compilation.logger.stderr = true;
                    self.profiling.logger.stderr = true;
                    self.autotune.logger.stderr = true;
                }
                "1" | "true" => {
                    let file_path = "/tmp/cubecl.log";
                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
                "0" | "false" => {
                    self.compilation.logger.level = CompilationLogLevel::Disabled;
                    self.profiling.logger.level = ProfilingLogLevel::Disabled;
                    self.autotune.logger.level = AutotuneLogLevel::Disabled;
                }
                file_path => {
                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
            }
        };

        if let Ok(val) = std::env::var("CUBECL_DEBUG_OPTION") {
            match val.as_str() {
                "debug" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Medium;
                    self.autotune.logger.level = AutotuneLogLevel::Full;
                }
                "debug-full" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Full;
                    self.autotune.logger.level = AutotuneLogLevel::Full;
                }
                "profile" => {
                    self.profiling.logger.level = ProfilingLogLevel::Basic;
                }
                "profile-medium" => {
                    self.profiling.logger.level = ProfilingLogLevel::Medium;
                }
                "profile-full" => {
                    self.profiling.logger.level = ProfilingLogLevel::Full;
                }
                _ => {}
            }
        };

        if let Ok(val) = std::env::var("CUBECL_AUTOTUNE_LEVEL") {
            match val.as_str() {
                "minimal" | "0" => {
                    self.autotune.level = AutotuneLevel::Minimal;
                }
                "balanced" | "1" => {
                    self.autotune.level = AutotuneLevel::Balanced;
                }
                "extensive" | "2" => {
                    self.autotune.level = AutotuneLevel::Extensive;
                }
                "full" | "3" => {
                    self.autotune.level = AutotuneLevel::Full;
                }
                _ => {}
            }
        }

        if let Ok(val) = std::env::var("CUBECL_CPU_F16_EVAL") {
            match val.as_str() {
                "per-operation" => {
                    self.compilation.f16_evaluation = Some(F16Evaluation::PerOperation);
                }
                "chain" => {
                    self.compilation.f16_evaluation = Some(F16Evaluation::Chain);
                }
                "accumulators" => {
                    self.compilation.f16_evaluation = Some(F16Evaluation::Accumulators);
                }
                _ => {}
            }
        }

        if let Ok(dir) = std::env::var("CUBECL_DEBUG_PLIRON") {
            self.compilation.dump_dir = Some(dir.into());
        }

        if let Some(enabled) = env_bool("CUBECL_TIME_PASSES") {
            self.compilation.time_passes = enabled;
        }

        if let Some(level) = env_variant("CUBECL_DEBUG_INFO") {
            self.compilation.debug_info = Some(level);
        }

        if let Some(format) = env_variant("CUBECL_SPIRV_DEBUG_FORMAT") {
            self.compilation.spirv_debug_format = format;
        }

        if let Ok(dir) = std::env::var("CUBECL_SOURCE_CACHE") {
            self.compilation.source_cache = Some(dir.into());
        }

        if let Some(enabled) = env_bool("CUBECL_THROUGHPUT_CACHE") {
            self.throughput.disable_cache = !enabled;
        }

        if let Ok(val) = std::env::var("CUBECL_ENVIRONMENT") {
            self.environment.name = val;
        }

        if let Ok(val) = std::env::var("CUBECL_ENVIRONMENT_RECORDS") {
            use cubecl_environment::records::RecordLevel;

            match val.as_str() {
                "off" | "0" => self.environment.records.level = RecordLevel::Off,
                "basic" | "1" => self.environment.records.level = RecordLevel::Basic,
                "full" | "2" => self.environment.records.level = RecordLevel::Full,
                _ => {}
            }
        }

        if let Some(enabled) = env_bool("CUBECL_AUTOTUNE_CACHE") {
            self.autotune.disable_cache = !enabled;
        }

        if let Some(enabled) = env_bool("CUBECL_AUTOTUNE_SHORT_CIRCUIT") {
            self.autotune.disable_short_circuit = !enabled;
        }

        if let Some(enabled) = env_bool("CUBECL_AUTOTUNE_BENCH_ADAPTIVE") {
            self.autotune.bench.adaptive = enabled;
        }

        if let Some(mode) = env_variant("CUBECL_DEVICE_TRANSPORT") {
            self.transport.device = mode;
        }

        self
    }
}
