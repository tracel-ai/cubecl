use super::logger::{LogLevel, LoggerConfig};

pub use cubecl_environment::stream::StreamPolicy;

/// Configuration for streaming settings in `CubeCL`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreamingConfig {
    /// Logger configuration for streaming logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<StreamingLogLevel>,
    /// The maximum number of streams to be used.
    #[serde(default = "default_max_streams")]
    pub max_streams: u8,
    /// Whether a batch that has filled up is waited for before more is queued.
    ///
    /// Not waiting hides the latency of preparing a launch, which is worth it
    /// when the launches are short enough that the device would otherwise sit
    /// idle between them. It costs memory: queued work holds its buffers, so a
    /// model with large activations pins them all at once for a gain it does
    /// not need. `Auto` measures the workload and picks; the rest force the
    /// choice. Ignored by backends whose flush does not wait.
    #[serde(default)]
    pub batch_wait: BatchWait,

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
            max_streams: default_max_streams(),
            batch_wait: BatchWait::default(),
            priority: StreamPriority::default(),
            policy: StreamPolicy::default(),
        }
    }
}

fn default_max_streams() -> u8 {
    128
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

/// Log levels for streaming in `CubeCL`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum StreamingLogLevel {
    /// Compilation logging is disabled.
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
        // Existing config files written before this field was added must
        // continue to deserialize unchanged.
        let cfg: StreamingConfig = toml::from_str("max_streams = 64").unwrap();
        assert_eq!(cfg.priority, StreamPriority::Default);
        assert_eq!(cfg.max_streams, 64);
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

    #[test]
    fn priority_serde_uses_lowercase_names() {
        // Stable on-disk representation — don't break user configs by accident.
        assert_eq!(
            serde_json::to_string(&StreamPriority::Low).unwrap(),
            "\"low\""
        );
        assert_eq!(
            serde_json::to_string(&StreamPriority::High).unwrap(),
            "\"high\""
        );
        assert_eq!(
            serde_json::to_string(&StreamPriority::Default).unwrap(),
            "\"default\""
        );
    }
}

/// Whether a filled batch is waited for before more work is queued behind it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BatchWait {
    /// Measure the workload and choose. A pool that spends its time waiting on
    /// the client wants the work queued behind it; one the client cannot keep
    /// up with gains nothing and would only hold the buffers.
    #[default]
    Auto,
    /// Wait for every filled batch.
    Always,
    /// Never wait for a filled batch.
    Never,
}
