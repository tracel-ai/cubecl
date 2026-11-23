use super::{LogLevel, LoggerConfig};
use cubecl_environment::stream::StreamPolicy;

/// How many streams a device runs at most, unless configured.
const DEFAULT_MAX_STREAM_COUNT: u8 = 128;

/// How many streams a device runs, at which priority, and how the current one is chosen.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreamingConfig {
    /// Where the streaming log goes, and how much it says.
    #[serde(default)]
    pub logger: LoggerConfig<StreamingLogLevel>,
    /// The most streams a device runs at once.
    #[serde(default = "default_max_stream_count")]
    pub max_stream_count: u8,
    /// Backend stream priority hint.
    ///
    /// Backends that expose stream priorities (e.g. CUDA via
    /// `cuStreamCreateWithPriority`) map this to their native range. Backends
    /// without a notion of stream priority ignore it. The default is
    /// [`StreamPriority::Default`], which preserves existing behavior.
    #[serde(default)]
    pub priority: StreamPriority,
    /// How the current stream is derived: `"per-thread"` (default),
    /// `"per-task"` (stable stream per async task, requires the `tokio`
    /// feature to take effect) or `"single"`.
    ///
    /// A programmatic `cubecl_environment::stream::set_policy` call always
    /// wins over this setting.
    #[serde(default)]
    pub policy: StreamPolicy,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            logger: Default::default(),
            max_stream_count: DEFAULT_MAX_STREAM_COUNT,
            priority: StreamPriority::default(),
            policy: StreamPolicy::default(),
        }
    }
}

fn default_max_stream_count() -> u8 {
    DEFAULT_MAX_STREAM_COUNT
}

/// Stream priority hint, mapped to the backend's native priority range.
///
/// CUDA convention is that lower numerical priorities run first; this enum is
/// the portable abstraction so other runtimes can adopt it without changing
/// their public surface. Backends clamp to their supported range — passing
/// [`StreamPriority::Low`] on a device whose only priority bucket is the
/// default is harmless.
///
/// The motivating use case is sharing a single GPU with a desktop compositor
/// (WSL2, dev laptops, single-GPU workstations): running long compute batches
/// on a low-priority stream lets the compositor preempt cleanly and keeps the
/// UI responsive.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StreamPriority {
    /// Default backend priority — current behavior on every backend.
    #[default]
    #[serde(rename = "default")]
    Default,
    /// Lowest priority the backend supports. Useful when sharing the GPU with
    /// an interactive workload such as a compositor; long-running compute
    /// batches yield to higher-priority work like UI rendering.
    #[serde(rename = "low")]
    Low,
    /// Highest priority the backend supports. Useful for latency-critical
    /// foreground work that must not be preempted by background batches.
    #[serde(rename = "high")]
    High,
}

/// How much the streaming log says.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum StreamingLogLevel {
    /// Streaming logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Basic streaming information is logged such as when streams are merged.
    #[serde(rename = "basic")]
    Basic,

    /// Full streaming details are logged.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for StreamingLogLevel {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_priority_is_default_variant() {
        assert_eq!(StreamingConfig::default().priority, StreamPriority::Default);
    }

    #[cfg(feature = "std")]
    #[test]
    fn priority_omitted_in_toml_falls_back_to_default() {
        let cfg: StreamingConfig = toml::from_str("max_stream_count = 64").unwrap();

        assert_eq!(cfg.priority, StreamPriority::Default);
    }

    #[cfg(feature = "std")]
    #[test]
    fn max_stream_count_is_read_from_toml() {
        let cfg: StreamingConfig = toml::from_str("max_stream_count = 64").unwrap();

        assert_eq!(cfg.max_stream_count, 64);
    }

    #[test]
    fn priority_serde_roundtrip() {
        for p in [
            StreamPriority::Default,
            StreamPriority::Low,
            StreamPriority::High,
        ] {
            let s = serde_json::to_string(&p).unwrap();
            let back: StreamPriority = serde_json::from_str(&s).unwrap();
            assert_eq!(p, back);
        }
    }

    /// Stable on-disk representation: don't break user configs by accident.
    #[test]
    fn priority_serde_uses_lowercase_names() {
        for (priority, name) in [
            (StreamPriority::Low, "\"low\""),
            (StreamPriority::High, "\"high\""),
            (StreamPriority::Default, "\"default\""),
        ] {
            assert_eq!(serde_json::to_string(&priority).unwrap(), name);
        }
    }
}
