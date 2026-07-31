use super::logger::{LogLevel, LoggerConfig};

/// Configuration for compilation settings in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    /// Logger configuration for compilation logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<CompilationLogLevel>,
    /// Whether compiled kernels are cached in the active environment.
    #[serde(default)]
    #[cfg(std_io)]
    pub cache: bool,
    /// Compile and tune, but dispatch nothing else.
    ///
    /// Every kernel is still compiled and cached, and autotune still runs and
    /// measures its candidates — those launches *are* the measurement. Every
    /// other launch is compiled and then dropped instead of reaching the
    /// device. This is what makes a warm-up pass cost compilation and tuning
    /// alone, rather than also running the workload that provokes them, and it
    /// is how a shippable environment is produced in a fraction of the time.
    ///
    /// **Buffers are left uninitialized**, so anything read back during such a
    /// pass is garbage. It is only meaningful for a pass that drives a workload
    /// for its *shapes* — shapes are what key the caches — and never branches
    /// on a computed value. Turning it on for real work produces wrong results,
    /// silently.
    ///
    /// Candidate timings are measured over uninitialized inputs too, so a
    /// kernel whose speed depends on its data (denormals, early exits) can tune
    /// differently than it would under a real workload.
    #[serde(default)]
    pub compile_only: bool,
    /// Controls whether kernel launches enforce bounds checks.
    #[serde(default)]
    pub check_mode: BoundsCheckMode,
}

/// Bounds checks options.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum BoundsCheckMode {
    #[serde(rename = "enforce")]
    /// Always enforce bounds checks on every kernel launch.
    Enforce,
    #[serde(rename = "validate")]
    /// Always enforce bounds checks on every kernel launch, and validate unchecked kernels for OOB.
    Validate,
    /// Enforce bounds checking on standard launches, but skip checks on
    /// explicitly unchecked launches for better performance.
    #[default]
    #[serde(rename = "auto")]
    Auto,
}

/// Log levels for compilation in `CubeCL`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompilationLogLevel {
    /// Compilation logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Basic compilation information is logged such as when kernels are compiled.
    #[serde(rename = "basic")]
    Basic,

    /// Full compilation details are logged including source code.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for CompilationLogLevel {}
