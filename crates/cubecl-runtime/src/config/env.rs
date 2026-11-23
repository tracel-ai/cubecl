use super::autotune::{AutotuneConfig, AutotuneLevel, AutotuneLogLevel};
use super::compilation::{CompilationConfig, CompilationLogLevel};
use super::profiling::{ProfilingConfig, ProfilingLogLevel};
use super::{CubeClRuntimeConfig, LogLevel, LoggerConfig};
use cubecl_environment::records::RecordLevel;
use std::path::PathBuf;

/// Where `CUBECL_DEBUG_LOG=1` writes the debug log.
const DEBUG_LOG_PATH: &str = "/tmp/cubecl.log";

/// `config` with every setting an environment variable overrides.
///
/// An unset variable, or one whose value names nothing, leaves its setting as configured.
pub fn override_from_env(config: CubeClRuntimeConfig) -> CubeClRuntimeConfig {
    let mut config = with_debug_option(with_debug_log(config));

    config.autotune.level =
        env_parsed("CUBECL_AUTOTUNE_LEVEL", autotune_level).unwrap_or(config.autotune.level);
    config.autotune.disable_cache =
        disabled_by("CUBECL_AUTOTUNE_CACHE", config.autotune.disable_cache);
    config.autotune.disable_short_circuit = disabled_by(
        "CUBECL_AUTOTUNE_SHORT_CIRCUIT",
        config.autotune.disable_short_circuit,
    );
    config.autotune.bench.adaptive =
        env_bool("CUBECL_AUTOTUNE_BENCH_ADAPTIVE").unwrap_or(config.autotune.bench.adaptive);

    config.compilation.f16_evaluation =
        env_variant("CUBECL_CPU_F16_EVAL").or(config.compilation.f16_evaluation);
    config.compilation.dump_dir = env_path("CUBECL_DEBUG_PLIRON").or(config.compilation.dump_dir);
    config.compilation.time_passes =
        env_bool("CUBECL_TIME_PASSES").unwrap_or(config.compilation.time_passes);
    config.compilation.debug_info =
        env_variant("CUBECL_DEBUG_INFO").or(config.compilation.debug_info);
    config.compilation.spirv_debug_format =
        env_variant("CUBECL_SPIRV_DEBUG_FORMAT").unwrap_or(config.compilation.spirv_debug_format);
    config.compilation.source_cache =
        env_path("CUBECL_SOURCE_CACHE").or(config.compilation.source_cache);

    config.throughput.disable_cache =
        disabled_by("CUBECL_THROUGHPUT_CACHE", config.throughput.disable_cache);

    config.environment.name =
        std::env::var("CUBECL_ENVIRONMENT").unwrap_or(config.environment.name);
    config.environment.records.level = env_parsed("CUBECL_ENVIRONMENT_RECORDS", record_level)
        .unwrap_or(config.environment.records.level);

    config.transport.device =
        env_variant("CUBECL_DEVICE_TRANSPORT").unwrap_or(config.transport.device);

    config
}

/// The variant of `T` that the environment variable `name` names with its `serde` name, as in
/// `cubecl.toml`. `None` when the variable is unset or names no variant, as for [`env_bool`].
fn env_variant<T: serde::de::DeserializeOwned>(name: &str) -> Option<T> {
    variant(&std::env::var(name).ok()?)
}

/// A boolean environment variable, or `None` when it is unset or unreadable.
///
/// An unrecognized value is `None` rather than an error: the variable is an
/// override, so failing to parse it means leaving the configured value alone.
fn env_bool(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.as_str() {
        "true" | "1" | "on" => Some(true),
        "false" | "0" | "off" => Some(false),
        _ => None,
    }
}

/// The variant of `T` with the `serde` name `value`.
fn variant<T: serde::de::DeserializeOwned>(value: &str) -> Option<T> {
    use serde::de::{IntoDeserializer, value::Error};

    T::deserialize(IntoDeserializer::<Error>::into_deserializer(value)).ok()
}

/// What `parse` reads from the environment variable `name`, or `None` when it is unset.
fn env_parsed<T>(name: &str, parse: fn(&str) -> Option<T>) -> Option<T> {
    parse(&std::env::var(name).ok()?)
}

/// The path the environment variable `name` holds, or `None` when it is unset.
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

/// Whether a cache stays disabled: the environment variable `name` enables or disables it, and
/// leaves `disabled` as it is when unset.
fn disabled_by(name: &str, disabled: bool) -> bool {
    env_bool(name).map_or(disabled, |enabled| !enabled)
}

/// The autotune level `CUBECL_AUTOTUNE_LEVEL` names, by name or by number.
fn autotune_level(value: &str) -> Option<AutotuneLevel> {
    match value {
        "minimal" | "0" => Some(AutotuneLevel::Minimal),
        "balanced" | "1" => Some(AutotuneLevel::Balanced),
        "extensive" | "2" => Some(AutotuneLevel::Extensive),
        "full" | "3" => Some(AutotuneLevel::Full),
        _ => None,
    }
}

/// The record level `CUBECL_ENVIRONMENT_RECORDS` names, by name or by number.
fn record_level(value: &str) -> Option<RecordLevel> {
    match value {
        "off" | "0" => Some(RecordLevel::Off),
        "basic" | "1" => Some(RecordLevel::Basic),
        "full" | "2" => Some(RecordLevel::Full),
        _ => None,
    }
}

/// `config` with what `CUBECL_DEBUG_LOG` asks for: the debug log levels, written to the sink it
/// names, or every debug log disabled.
fn with_debug_log(config: CubeClRuntimeConfig) -> CubeClRuntimeConfig {
    let Ok(value) = std::env::var("CUBECL_DEBUG_LOG") else {
        return config;
    };
    let config = with_debug_levels(config, DebugLevels::DEBUG);
    match value.as_str() {
        "stdout" => with_debug_sink(config, &DebugSink::Stdout),
        "stderr" => with_debug_sink(config, &DebugSink::Stderr),
        "1" | "true" => with_debug_sink(config, &DebugSink::File(DEBUG_LOG_PATH.into())),
        "0" | "false" => with_debug_levels(config, DebugLevels::DISABLED),
        path => with_debug_sink(config, &DebugSink::File(path.into())),
    }
}

/// `config` with the log levels `CUBECL_DEBUG_OPTION` names.
fn with_debug_option(config: CubeClRuntimeConfig) -> CubeClRuntimeConfig {
    let Ok(value) = std::env::var("CUBECL_DEBUG_OPTION") else {
        return config;
    };
    match value.as_str() {
        "debug" => with_debug_levels(config, DebugLevels::DEBUG),
        "debug-full" => with_debug_levels(config, DebugLevels::DEBUG_FULL),
        "profile" => with_profiling_level(config, ProfilingLogLevel::Basic),
        "profile-medium" => with_profiling_level(config, ProfilingLogLevel::Medium),
        "profile-full" => with_profiling_level(config, ProfilingLogLevel::Full),
        _ => config,
    }
}

/// `config` with the compilation, profiling and autotune logs at `levels`.
fn with_debug_levels(config: CubeClRuntimeConfig, levels: DebugLevels) -> CubeClRuntimeConfig {
    CubeClRuntimeConfig {
        compilation: CompilationConfig {
            logger: LoggerConfig {
                level: levels.compilation,
                ..config.compilation.logger
            },
            ..config.compilation
        },
        profiling: ProfilingConfig {
            logger: LoggerConfig {
                level: levels.profiling,
                ..config.profiling.logger
            },
        },
        autotune: AutotuneConfig {
            logger: LoggerConfig {
                level: levels.autotune,
                ..config.autotune.logger
            },
            ..config.autotune
        },
        ..config
    }
}

/// `config` with the profiling log at `level`.
fn with_profiling_level(
    config: CubeClRuntimeConfig,
    level: ProfilingLogLevel,
) -> CubeClRuntimeConfig {
    CubeClRuntimeConfig {
        profiling: ProfilingConfig {
            logger: LoggerConfig {
                level,
                ..config.profiling.logger
            },
        },
        ..config
    }
}

/// `config` with the compilation, profiling and autotune logs also written to `sink`.
fn with_debug_sink(config: CubeClRuntimeConfig, sink: &DebugSink) -> CubeClRuntimeConfig {
    CubeClRuntimeConfig {
        compilation: CompilationConfig {
            logger: sink.add_to(config.compilation.logger),
            ..config.compilation
        },
        profiling: ProfilingConfig {
            logger: sink.add_to(config.profiling.logger),
        },
        autotune: AutotuneConfig {
            logger: sink.add_to(config.autotune.logger),
            ..config.autotune
        },
        ..config
    }
}

/// The levels the debug variables set the compilation, profiling and autotune logs to together.
struct DebugLevels {
    pub compilation: CompilationLogLevel,
    pub profiling: ProfilingLogLevel,
    pub autotune: AutotuneLogLevel,
}

impl DebugLevels {
    /// What `CUBECL_DEBUG_LOG` and `CUBECL_DEBUG_OPTION=debug` turn on.
    const DEBUG: Self = Self {
        compilation: CompilationLogLevel::Full,
        profiling: ProfilingLogLevel::Medium,
        autotune: AutotuneLogLevel::Full,
    };
    /// What `CUBECL_DEBUG_OPTION=debug-full` turns on.
    const DEBUG_FULL: Self = Self {
        compilation: CompilationLogLevel::Full,
        profiling: ProfilingLogLevel::Full,
        autotune: AutotuneLogLevel::Full,
    };
    /// What `CUBECL_DEBUG_LOG=0` leaves.
    const DISABLED: Self = Self {
        compilation: CompilationLogLevel::Disabled,
        profiling: ProfilingLogLevel::Disabled,
        autotune: AutotuneLogLevel::Disabled,
    };
}

/// Where `CUBECL_DEBUG_LOG` writes the debug logs.
enum DebugSink {
    Stdout,
    Stderr,
    File(PathBuf),
}

impl DebugSink {
    /// `logger`, also writing to this sink.
    fn add_to<L: LogLevel>(&self, logger: LoggerConfig<L>) -> LoggerConfig<L> {
        match self {
            Self::Stdout => LoggerConfig {
                stdout: true,
                ..logger
            },
            Self::Stderr => LoggerConfig {
                stderr: true,
                ..logger
            },
            Self::File(path) => LoggerConfig {
                file: Some(path.clone()),
                ..logger
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_autotune_level_is_named_by_name_or_number() {
        assert!(matches!(
            [autotune_level("extensive"), autotune_level("2")],
            [
                Some(AutotuneLevel::Extensive),
                Some(AutotuneLevel::Extensive)
            ]
        ));
    }

    #[test]
    fn an_unknown_autotune_level_is_none() {
        assert!(autotune_level("fast").is_none());
    }

    #[test]
    fn a_record_level_is_named_by_name_or_number() {
        assert_eq!(
            [record_level("full"), record_level("2")],
            [Some(RecordLevel::Full), Some(RecordLevel::Full)]
        );
    }

    #[test]
    fn debug_levels_set_the_three_logs() {
        let config = with_debug_levels(CubeClRuntimeConfig::default(), DebugLevels::DEBUG_FULL);

        assert!(matches!(
            (
                config.compilation.logger.level,
                config.profiling.logger.level,
                config.autotune.logger.level
            ),
            (
                CompilationLogLevel::Full,
                ProfilingLogLevel::Full,
                AutotuneLogLevel::Full
            )
        ));
    }

    #[test]
    fn a_profiling_level_leaves_the_other_logs() {
        let config = with_profiling_level(
            with_debug_levels(CubeClRuntimeConfig::default(), DebugLevels::DISABLED),
            ProfilingLogLevel::Basic,
        );

        assert!(matches!(
            config.compilation.logger.level,
            CompilationLogLevel::Disabled
        ));
    }

    #[test]
    fn a_debug_sink_is_added_to_the_three_logs() {
        let config = with_debug_sink(CubeClRuntimeConfig::default(), &DebugSink::Stderr);

        assert!(
            config.compilation.logger.stderr
                && config.profiling.logger.stderr
                && config.autotune.logger.stderr
        );
    }

    #[test]
    fn a_file_sink_keeps_the_other_sinks() {
        let logger = LoggerConfig::<ProfilingLogLevel> {
            stdout: true,
            ..Default::default()
        };

        let logger = DebugSink::File(DEBUG_LOG_PATH.into()).add_to(logger);

        assert!(logger.stdout && logger.file == Some(DEBUG_LOG_PATH.into()));
    }
}
