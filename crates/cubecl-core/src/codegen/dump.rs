use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use pliron::pass::PMConfig;

/// The compiler dump of one kernel: its IR after every pass, plus the files each compiler adds.
///
/// Enabled when `compilation.dump_dir` (or `CUBECL_DEBUG_PLIRON`) is set. Otherwise every method
/// is a no-op, so compilers call them unconditionally.
#[derive(Clone, Debug, Default)]
pub struct KernelDump {
    #[cfg(feature = "std")]
    dir: Option<std::path::PathBuf>,
}

impl KernelDump {
    /// The dump of `kernel_name`, in a subdirectory of the configured dump directory.
    #[must_use]
    #[cfg_attr(not(feature = "std"), allow(unused_variables))]
    pub fn new(kernel_name: &str) -> Self {
        Self {
            #[cfg(feature = "std")]
            dir: kernel_dir(kernel_name),
        }
    }

    /// Whether this kernel is dumped.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        #[cfg(feature = "std")]
        return self.dir.is_some();
        #[cfg(not(feature = "std"))]
        false
    }

    /// Writes `file_name` to the dump. `contents` runs only when the dump is enabled.
    #[cfg_attr(not(feature = "std"), allow(unused_variables))]
    pub fn write<C: AsRef<[u8]>>(&self, file_name: &str, contents: impl FnOnce() -> C) {
        #[cfg(feature = "std")]
        if let Some(dir) = &self.dir {
            let path = dir.join(file_name);
            if let Err(err) = std::fs::write(&path, contents()) {
                log::warn!("Can't write the dump file {}: {err}", path.display());
            }
        }
    }

    /// Pass manager settings: print the IR after each pass to the dump, and time the passes
    /// when `compilation.time_passes` is set.
    #[must_use]
    pub fn pass_config(&self) -> PMConfig {
        PMConfig {
            print_after_all: self.is_enabled(),
            #[cfg(feature = "std")]
            ir_printing_dir: self.dir.clone(),
            time_all_passes: CubeClRuntimeConfig::get().compilation.time_passes,
            ..Default::default()
        }
    }
}

/// Creates and returns the dump directory of `kernel_name`, if dumps are enabled.
#[cfg(feature = "std")]
fn kernel_dir(kernel_name: &str) -> Option<std::path::PathBuf> {
    let root = CubeClRuntimeConfig::get().compilation.dump_dir.clone()?;
    create_kernel_dir(&root, kernel_name)
}

#[cfg(feature = "std")]
fn create_kernel_dir(root: &std::path::Path, kernel_name: &str) -> Option<std::path::PathBuf> {
    let options = sanitize_filename::Options {
        replacement: "_",
        ..Default::default()
    };
    let dir = root.join(sanitize_filename::sanitize_with_options(
        kernel_name,
        options,
    ));
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(dir),
        Err(err) => {
            log::warn!("Can't create the dump directory {}: {err}", dir.display());
            None
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn disabled_dump_builds_no_contents() {
        KernelDump::default().write("never.txt", || -> &[u8] {
            panic!("contents must not be built")
        });
    }

    #[test]
    fn enabled_dump_writes_to_a_sanitized_kernel_dir() {
        let root = std::env::temp_dir().join(alloc::format!("cubecl-dump-{}", std::process::id()));
        let dump = KernelDump {
            dir: create_kernel_dir(&root, "matmul<f32>/tile"),
        };
        dump.write("module.cpp", || "kernel");

        let dir = root.join("matmul_f32__tile");
        assert_eq!(
            std::fs::read_to_string(dir.join("module.cpp")).unwrap(),
            "kernel"
        );
        let config = dump.pass_config();
        assert!(config.print_after_all);
        assert_eq!(config.ir_printing_dir, Some(dir));
        std::fs::remove_dir_all(root).unwrap();
    }
}
