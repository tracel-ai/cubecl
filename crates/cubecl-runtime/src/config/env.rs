use super::autotune::{AutotuneConfig, AutotuneLevel, AutotuneLogLevel};
use super::compilation::{CompilationConfig, CompilationLogLevel};
use super::{CubeClRuntimeConfig, LogLevel, LoggerConfig};
use cubecl_environment::records::RecordLevel;
use std::path::PathBuf;

/// Where `CUBECL_DEBUG_LOG=1` writes the debug log.
const DEBUG_LOG_PATH: &str = "/tmp/cubecl.log";

/// `config` with every setting an environment variable overrides.
///
/// An unset variable, or one whose value names nothing, leaves its setting as configured.
pub fn override_from_env(config: CubeClRuntimeConfig) -> CubeClRuntimeConfig {
    let config = with_debug_log(config, std::env::var("CUBECL_DEBUG_LOG").ok().as_deref());
    let mut config =
        with_debug_option(config, std::env::var("CUBECL_DEBUG_OPTION").ok().as_deref());

    config.autotune.level =
        env_parsed("CUBECL_AUTOTUNE_LEVEL", autotune_level).unwrap_or(config.autotune.level);
    config.autotune.disable_cache = disabled_by(
        env_bool("CUBECL_AUTOTUNE_CACHE"),
        config.autotune.disable_cache,
    );
    config.autotune.disable_short_circuit = disabled_by(
        env_bool("CUBECL_AUTOTUNE_SHORT_CIRCUIT"),
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

    config.throughput.disable_cache = disabled_by(
        env_bool("CUBECL_THROUGHPUT_CACHE"),
        config.throughput.disable_cache,
    );

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
    env_parsed(name, variant)
}

/// A boolean environment variable, or `None` when it is unset, unreadable or unrecognized.
fn env_bool(name: &str) -> Option<bool> {
    env_parsed(name, parse_bool)
}

/// The boolean `value` spells. An unrecognized value is `None` rather than an error: the
/// variable is an override, so failing to parse it means leaving the configured value alone.
fn parse_bool(value: &str) -> Option<bool> {
    match value {
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

/// Whether a cache is disabled once a variable `enabled` it or not, or `disabled` when the
/// variable is unset.
fn disabled_by(enabled: Option<bool>, disabled: bool) -> bool {
    enabled.map_or(disabled, |enabled| !enabled)
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

/// `config` with what `CUBECL_DEBUG_LOG` set to `value` asks for: the debug log levels, written
/// to the sink it names, or every debug log disabled.
fn with_debug_log(config: CubeClRuntimeConfig, value: Option<&str>) -> CubeClRuntimeConfig {
    let Some(value) = value else {
        return config;
    };
    let config = with_debug_levels(config, DebugLevels::DEBUG);
    match value {
        "stdout" => with_debug_sink(config, &DebugSink::Stdout),
        "stderr" => with_debug_sink(config, &DebugSink::Stderr),
        "1" | "true" => with_debug_sink(config, &DebugSink::File(DEBUG_LOG_PATH.into())),
        "0" | "false" => with_debug_levels(config, DebugLevels::DISABLED),
        path => with_debug_sink(config, &DebugSink::File(path.into())),
    }
}

/// `config` with the log levels `CUBECL_DEBUG_OPTION` set to `value` names.
fn with_debug_option(config: CubeClRuntimeConfig, value: Option<&str>) -> CubeClRuntimeConfig {
    let Some(value) = value else {
        return config;
    };
    match value {
        "debug" | "debug-full" => with_debug_levels(config, DebugLevels::DEBUG),
        _ => config,
    }
}

/// `config` with the compilation and autotune logs at `levels`.
fn with_debug_levels(config: CubeClRuntimeConfig, levels: DebugLevels) -> CubeClRuntimeConfig {
    CubeClRuntimeConfig {
        compilation: CompilationConfig {
            logger: LoggerConfig {
                level: levels.compilation,
                ..config.compilation.logger
            },
            ..config.compilation
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

/// `config` with the compilation and autotune logs also written to `sink`.
fn with_debug_sink(config: CubeClRuntimeConfig, sink: &DebugSink) -> CubeClRuntimeConfig {
    CubeClRuntimeConfig {
        compilation: CompilationConfig {
            logger: sink.add_to(config.compilation.logger),
            ..config.compilation
        },
        autotune: AutotuneConfig {
            logger: sink.add_to(config.autotune.logger),
            ..config.autotune
        },
        ..config
    }
}

/// The levels the debug variables set the compilation and autotune logs to together.
struct DebugLevels {
    pub compilation: CompilationLogLevel,
    pub autotune: AutotuneLogLevel,
}

impl DebugLevels {
    /// What `CUBECL_DEBUG_LOG` and `CUBECL_DEBUG_OPTION=debug` or `debug-full` turn on.
    const DEBUG: Self = Self {
        compilation: CompilationLogLevel::Full,
        autotune: AutotuneLogLevel::Full,
    };
    /// What `CUBECL_DEBUG_LOG=0` leaves.
    const DISABLED: Self = Self {
        compilation: CompilationLogLevel::Disabled,
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

    fn default() -> CubeClRuntimeConfig {
        CubeClRuntimeConfig::default()
    }

    #[test]
    fn on_spells_true() {
        assert_eq!(parse_bool("on"), Some(true));
    }

    #[test]
    fn off_spells_false() {
        assert_eq!(parse_bool("off"), Some(false));
    }

    #[test]
    fn an_unknown_boolean_is_none() {
        assert_eq!(parse_bool("maybe"), None);
    }

    #[test]
    fn a_variant_is_read_by_its_serde_name() {
        assert_eq!(variant::<RecordLevel>("full"), Some(RecordLevel::Full));
    }

    #[test]
    fn enabling_a_cache_clears_disabled() {
        assert!(!disabled_by(Some(true), true));
    }

    #[test]
    fn an_unset_variable_keeps_disabled() {
        assert!(disabled_by(None, true));
    }

    #[test]
    fn an_autotune_level_is_named_by_name() {
        assert!(matches!(
            autotune_level("extensive"),
            Some(AutotuneLevel::Extensive)
        ));
    }

    #[test]
    fn an_autotune_level_is_named_by_number() {
        assert!(matches!(
            autotune_level("2"),
            Some(AutotuneLevel::Extensive)
        ));
    }

    #[test]
    fn an_unknown_autotune_level_is_none() {
        assert!(autotune_level("fast").is_none());
    }

    #[test]
    fn a_record_level_is_named_by_name() {
        assert_eq!(record_level("full"), Some(RecordLevel::Full));
    }

    #[test]
    fn a_record_level_is_named_by_number() {
        assert_eq!(record_level("2"), Some(RecordLevel::Full));
    }

    #[test]
    fn an_unset_debug_log_changes_nothing() {
        let config = with_debug_log(default(), None);

        assert!(matches!(
            config.compilation.logger.level,
            CompilationLogLevel::Disabled
        ));
    }

    #[test]
    fn a_debug_log_turned_off_disables_the_three_logs() {
        let config = with_debug_log(default(), Some("0"));

        assert!(matches!(
            config.autotune.logger.level,
            AutotuneLogLevel::Disabled
        ));
    }

    #[test]
    fn a_debug_log_path_becomes_a_file_sink() {
        let config = with_debug_log(default(), Some("/var/log/cube.log"));

        assert_eq!(
            config.compilation.logger.file,
            Some(PathBuf::from("/var/log/cube.log"))
        );
    }

    #[test]
    fn the_debug_full_option_sets_the_debug_levels() {
        let config = with_debug_option(default(), Some("debug-full"));

        assert!(matches!(
            config.compilation.logger.level,
            CompilationLogLevel::Full
        ));
    }

    #[test]
    fn an_unknown_debug_option_changes_nothing() {
        let config = with_debug_option(default(), Some("loud"));

        assert!(matches!(
            config.compilation.logger.level,
            CompilationLogLevel::Disabled
        ));
    }

    #[test]
    fn debug_levels_set_both_logs() {
        let config = with_debug_levels(default(), DebugLevels::DEBUG);

        assert!(matches!(
            (
                config.compilation.logger.level,
                config.autotune.logger.level
            ),
            (CompilationLogLevel::Full, AutotuneLogLevel::Full)
        ));
    }

    #[test]
    fn a_debug_sink_is_added_to_both_logs() {
        let config = with_debug_sink(default(), &DebugSink::Stderr);

        assert!(config.compilation.logger.stderr && config.autotune.logger.stderr);
    }

    #[test]
    fn a_file_sink_keeps_the_other_sinks() {
        let logger = LoggerConfig::<CompilationLogLevel> {
            stdout: true,
            ..Default::default()
        };

        let logger = DebugSink::File(DEBUG_LOG_PATH.into()).add_to(logger);

        assert!(logger.stdout && logger.file == Some(DEBUG_LOG_PATH.into()));
    }
}
