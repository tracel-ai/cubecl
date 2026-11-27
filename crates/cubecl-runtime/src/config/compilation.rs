use super::debug_info::resolve_debug_info;
use super::{LogLevel, LoggerConfig};
use cubecl_ir::settings::DebugInfo;

/// The debug data the cargo profile asks for: [`LineTables`](DebugInfo::LineTables) when the
/// profile sets `debug`, as `dev` does by default, else [`None`](DebugInfo::None).
pub const PROFILE_DEBUG_INFO: DebugInfo = if cfg!(cubecl_debug_info) {
    DebugInfo::LineTables
} else {
    DebugInfo::None
};

/// How kernels compile: their bounds checks, debug data and dumps, and the threads they compile on.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    /// Where the compilation log goes, and how much it says.
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
    /// [`LaunchMode::Queue`](crate::execution::LaunchMode::Queue). The
    /// queue itself is as long as what was queued; each thread takes the next
    /// kernel as it finishes one. `None` uses every core the process may run
    /// on; a smaller number leaves cores to other work. It does not bound
    /// memory: a batch holds every queued kernel's artifacts until it ends.
    #[serde(default)]
    pub parallelism: Option<usize>,
    /// Whether a kernel whose compiled code spills registers to scratch memory fails to compile
    /// with [`CompilationError::RegisterSpill`](crate::compiler::CompilationError::RegisterSpill).
    /// An autotune then drops the candidate as it drops any that fails to compile, and a launch
    /// of the kernel fails. Off by default: a kernel with no candidate that fits in registers
    /// would fail where it now runs slowly. Set by `CUBECL_REFUSE_REGISTER_SPILLS`.
    ///
    /// HIP reports spills for both of its backends, from the code object's metadata, and CUDA
    /// from the assembler's log. wgpu, Metal and the CPU runtime report none, so they never
    /// refuse a kernel, and neither does CUDA for a kernel its driver assembled when it loaded.
    #[serde(default)]
    pub refuse_register_spills: bool,
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
            .unwrap_or_else(|| cubecl_common::device_handle::ProcessCpus::get().parallelism());
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

/// Which launches check their accesses against the bounds of what they read and write.
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

/// How much the compilation log says.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum CompilationLogLevel {
    /// No compilation log is written.
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
    fn the_configured_limit_lowers_debug_data() {
        let config = CompilationConfig {
            debug_info: Some(DebugInfo::LineTables),
            ..Default::default()
        };

        assert_eq!(
            config.resolve_debug_info(DebugInfo::Full),
            DebugInfo::LineTables
        );
    }

    #[test]
    fn at_least_one_thread_compiles() {
        let config = CompilationConfig {
            parallelism: Some(0),
            ..Default::default()
        };

        assert_eq!(config.parallelism(), 1);
    }

    #[test]
    fn a_host_with_native_f16_rounds_per_operation() {
        assert_eq!(
            F16Evaluation::for_native_f16(true),
            F16Evaluation::PerOperation
        );
    }

    #[test]
    fn a_host_without_native_f16_rounds_chains() {
        assert_eq!(F16Evaluation::for_native_f16(false), F16Evaluation::Chain);
    }

    #[cfg(feature = "std")]
    #[test]
    fn register_spills_are_accepted_by_default() {
        let config: crate::config::CubeClRuntimeConfig = toml::from_str("").unwrap();

        assert!(!config.compilation.refuse_register_spills);
    }

    #[cfg(feature = "std")]
    #[test]
    fn register_spills_can_be_refused() {
        let config: crate::config::CubeClRuntimeConfig =
            toml::from_str("[compilation]\nrefuse_register_spills = true").unwrap();

        assert!(config.compilation.refuse_register_spills);
    }

    #[cfg(feature = "std")]
    #[test]
    fn display_writes_the_config_name() {
        for mode in [
            F16Evaluation::PerOperation,
            F16Evaluation::Chain,
            F16Evaluation::Accumulators,
        ] {
            assert_eq!(
                serde_json::to_string(&mode).unwrap(),
                alloc::format!("\"{mode}\"")
            );
        }
    }
}
