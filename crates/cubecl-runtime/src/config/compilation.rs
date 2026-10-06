use super::logger::{LogLevel, LoggerConfig};

/// Configuration for compilation settings in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    /// Logger configuration for compilation logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<CompilationLogLevel>,
    /// Whether compiled kernels are cached in the active environment.
    #[serde(default)]
    #[cfg(persistence)]
    pub cache: bool,
    /// Controls whether kernel launches enforce bounds checks.
    #[serde(default)]
    pub check_mode: BoundsCheckMode,
    /// How far the CPU runtime carries an f16 intermediate in f32 before rounding it. `None`
    /// chooses by whether the host computes in f16 directly. Other runtimes ignore it.
    #[serde(default)]
    pub f16_evaluation: Option<F16Evaluation>,
    /// Directory for compiler dumps. Each kernel writes its IR after every compiler pass, and
    /// its final source, to a subdirectory named after the kernel. Set by `CUBECL_DEBUG_PLIRON`.
    #[serde(default)]
    #[cfg(feature = "std")]
    pub dump_dir: Option<std::path::PathBuf>,
    /// Log how long each compiler pass takes, at the `info` level. Set by `CUBECL_TIME_PASSES`.
    #[serde(default)]
    pub time_passes: bool,
}

/// How far an f32 intermediate is allowed to travel before it is rounded back to f16.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum F16Evaluation {
    /// Round after every operation, which is what a GPU does. Where f16 is not native it costs a
    /// convert pair per operation.
    #[serde(rename = "per-operation")]
    PerOperation,
    /// Round where a value is stored, a `let mut` included, or read by anything but arithmetic.
    /// The default where f16 is not native.
    #[default]
    #[serde(rename = "chain")]
    Chain,
    /// Also hold a private f16 variable in f32 where that removes more converts than it adds, so
    /// a running total read often enough inside its loop stays in f16. Costs vector registers, so
    /// a wide kernel may want a narrower line.
    #[serde(rename = "accumulators")]
    Accumulators,
}

impl F16Evaluation {
    /// The mode for a host that does or does not compute in f16 directly. Where it does, a chain
    /// held in f32 only adds converts.
    pub fn for_native_f16(native: bool) -> Self {
        match native {
            true => Self::PerOperation,
            false => Self::Chain,
        }
    }
}

impl core::fmt::Display for F16Evaluation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::PerOperation => "per-operation",
            Self::Chain => "chain",
            Self::Accumulators => "accumulators",
        })
    }
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
