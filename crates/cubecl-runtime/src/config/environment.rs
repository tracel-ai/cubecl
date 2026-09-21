use alloc::string::{String, ToString};

#[cfg(std_io)]
use super::cache::CacheConfig;

/// Which named environment the process warms into, and where environments are
/// kept.
///
/// An environment is one local store holding every cached namespace: autotune
/// results, compiled kernels, throughput measurements. Exactly one is active
/// at a time, so naming them lets a single checkout keep several side by side.
///
/// ```toml
/// [environment]
/// name = "h100"
/// path = "target"
///
/// [environment.records]
/// level = "basic"
/// keep_sessions = 8
/// ```
///
/// `CUBECL_ENVIRONMENT` overrides the name.
///
/// `path` only exists where there is a file system to put environments on;
/// elsewhere the name still selects one, it just isn't backed by a directory.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentConfig {
    /// Name of the environment to activate when the configuration loads.
    #[serde(default = "default_name")]
    pub name: String,

    /// Directory the environments live in.
    #[cfg(std_io)]
    #[serde(default)]
    pub path: CacheConfig,

    /// What the environment remembers of how it was built.
    #[serde(default)]
    pub records: RecordsConfig,
}

/// How much an environment records of its own build; see
/// [`cubecl_environment::records`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct RecordsConfig {
    /// `basic` by default: every record, without the heavy parts.
    #[serde(default)]
    pub level: cubecl_environment::records::RecordLevel,

    /// How many sessions an environment keeps; the oldest beyond it are
    /// pruned when a session starts. Every session is kept when unset.
    #[serde(default)]
    pub keep_sessions: Option<u32>,
}

fn default_name() -> String {
    cubecl_environment::environment::DEFAULT.to_string()
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            name: default_name(),
            #[cfg(std_io)]
            path: CacheConfig::default(),
            records: RecordsConfig::default(),
        }
    }
}
