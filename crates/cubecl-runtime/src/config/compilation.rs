use super::logger::{LogLevel, LoggerConfig};
use cubecl_ir::settings::DebugInfo;

/// The debug data the cargo profile asks for: [`LineTables`](DebugInfo::LineTables) when the
/// profile sets `debug`, as `dev` does by default, else [`None`](DebugInfo::None).
pub const PROFILE_DEBUG_INFO: DebugInfo = if cfg!(cubecl_debug_info) {
    DebugInfo::LineTables
} else {
    DebugInfo::None
};

/// The debug data of a kernel that asks for `requested`, with the global configuration: the same
/// level as [`CompilationConfig::resolve_debug_info`]. It reads the configuration once, so a kernel
/// id can call it at each launch.
pub fn effective_debug_info(requested: DebugInfo) -> DebugInfo {
    use super::{CubeClRuntimeConfig, RuntimeConfig};
    use core::sync::atomic::{AtomicU8, Ordering};

    const UNREAD: u8 = u8::MAX;
    const NO_LIMIT: u8 = u8::MAX - 1;
    static LIMIT: AtomicU8 = AtomicU8::new(UNREAD);

    let mut limit = LIMIT.load(Ordering::Relaxed);
    if limit == UNREAD {
        limit = CubeClRuntimeConfig::get()
            .compilation
            .debug_info
            .map_or(NO_LIMIT, |level| level as u8);
        LIMIT.store(limit, Ordering::Relaxed);
    }
    // Decode with the casts that encoded the level, so the order of the variants does not matter.
    let limit = [DebugInfo::None, DebugInfo::LineTables, DebugInfo::Full]
        .into_iter()
        .find(|level| *level as u8 == limit);
    resolve_debug_info(requested, limit)
}

/// At least [`PROFILE_DEBUG_INFO`], at most `limit`.
fn resolve_debug_info(requested: DebugInfo, limit: Option<DebugInfo>) -> DebugInfo {
    let level = requested.max(PROFILE_DEBUG_INFO);
    limit.map_or(level, |limit| level.min(limit))
}

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
    /// The most debug data a kernel may carry. `None` keeps what the cargo profile and the kernel
    /// ask for. It can only lower that level. Set by `CUBECL_DEBUG_INFO`.
    #[serde(default)]
    pub debug_info: Option<DebugInfo>,
    /// The format of the debug data in SPIR-V kernels. Set by `CUBECL_SPIRV_DEBUG_FORMAT`.
    #[serde(default)]
    pub spirv_debug_format: SpirvDebugFormat,
    /// A directory for the source text of kernels with full debug data. When no directory on this
    /// computer has the source files of a kernel, cubecl writes the texts into this directory, and
    /// the debug data points to them. `None` writes no files. Set by `CUBECL_SOURCE_CACHE`.
    #[serde(default)]
    #[cfg(feature = "std")]
    pub source_cache: Option<std::path::PathBuf>,
    /// How many threads a server compiles a queue of kernels on — see
    /// [`LaunchMode::CompileOnly`](crate::dry_run::LaunchMode::CompileOnly). The
    /// queue itself is as long as what was queued; each thread takes the next
    /// kernel as it finishes one. `None` uses every core the process may run
    /// on; a smaller number leaves cores to other work. It does not bound
    /// memory: a batch holds every queued kernel's artifacts until it ends.
    #[serde(default)]
    pub parallelism: Option<usize>,
}

impl CompilationConfig {
    /// The debug data of a kernel that asks for `requested`: at least [`PROFILE_DEBUG_INFO`], at
    /// most [`debug_info`](Self::debug_info).
    #[must_use]
    pub fn resolve_debug_info(&self, requested: DebugInfo) -> DebugInfo {
        resolve_debug_info(requested, self.debug_info)
    }

    /// How many threads compile at once: the configured count, or every core
    /// the process may run on. Never zero, and one without threads — on wasm
    /// too, whatever is configured, since it cannot spawn them.
    pub fn parallelism(&self) -> usize {
        #[cfg(all(feature = "std", not(target_family = "wasm")))]
        let threads = self
            .parallelism
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |cores| cores.get()));
        #[cfg(any(not(feature = "std"), target_family = "wasm"))]
        let threads = 1;

        threads.max(1)
    }
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

/// The format of the debug data in SPIR-V kernels. A kernel without debug data ignores it.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpirvDebugFormat {
    /// `NonSemantic` when the device supports it, else `OpLine`.
    #[default]
    #[serde(rename = "auto")]
    Auto,
    /// Core `OpLine`: the line of the innermost `#[cube]` function only. All devices support it.
    #[serde(rename = "op-line")]
    OpLine,
    /// `NonSemantic.Shader.DebugInfo.100`: each inlined `#[cube]` function is a separate frame.
    /// A device without support gets `OpLine`.
    #[serde(rename = "non-semantic")]
    NonSemantic,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_info_follows_the_profile_and_the_kernel() {
        let config = CompilationConfig::default();
        assert_eq!(
            config.resolve_debug_info(DebugInfo::None),
            PROFILE_DEBUG_INFO
        );
        assert_eq!(config.resolve_debug_info(DebugInfo::Full), DebugInfo::Full);
    }

    #[test]
    fn debug_info_limit_only_lowers() {
        let config = CompilationConfig {
            debug_info: Some(DebugInfo::LineTables),
            ..Default::default()
        };
        assert_eq!(
            config.resolve_debug_info(DebugInfo::Full),
            DebugInfo::LineTables
        );

        let config = CompilationConfig {
            debug_info: Some(DebugInfo::Full),
            ..Default::default()
        };
        assert_eq!(
            config.resolve_debug_info(DebugInfo::None),
            PROFILE_DEBUG_INFO
        );
    }
}
