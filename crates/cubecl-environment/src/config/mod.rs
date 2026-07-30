/// Reusable logger configuration and sink management.
pub mod logger;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Reports a configuration file that exists but could not be read into the
/// configuration type.
///
/// Goes to stderr as well as `log`. A malformed file is silently replaced by
/// [`Default`], and configuration is resolved on first use — typically before
/// the application has installed a `log` sink, so a `log`-only warning is
/// usually written to nothing at all. That combination is how a single stale
/// field turns every setting in a file into its default with nothing to notice.
///
/// Only reached when a file is present *and* malformed, which is always a
/// mistake worth interrupting for. A missing file, or one without the section
/// being looked for, is ordinary and stays quiet.
#[cfg(std_io)]
fn report_malformed(message: &str) {
    log::warn!("{message}");
    std::eprintln!("cubecl config: {message}");
}

/// Trait for runtime configurations potentially loaded from a TOML file.
///
/// Implementors provide a global storage slot and the set of file names to search for;
/// the trait supplies the lookup, lazy-initialization, and serialization logic.
///
/// The singleton stored in [`Config::storage`] is initialized on the first call to
/// [`Config::get`] by walking up the current working directory looking for any of the
/// names returned by [`Config::file_names`]. If none is found, [`Default`] is used.
pub trait RuntimeConfig:
    Default + Clone + Serialize + DeserializeOwned + Send + Sync + 'static
{
    /// Global storage for the configuration singleton.
    ///
    /// Each implementor must declare its own `static` slot, because Rust traits
    /// cannot own statics directly.
    fn storage() -> &'static crate::sync::Mutex<Option<Arc<Self>>>;

    /// File names searched in each directory during [`Config::from_current_dir`].
    ///
    /// The first existing file wins.
    fn file_names() -> &'static [&'static str];

    /// File names searched in each directory, where only a specific TOML section is loaded
    /// instead of the whole file.
    ///
    /// Each entry is `(file_name, section_name)` and the section must deserialize to `Self`.
    /// Checked after [`Config::file_names`] at each directory level.
    fn section_file_names() -> &'static [(&'static str, &'static str)] {
        &[]
    }

    /// Hook to override fields from environment variables after loading from disk.
    ///
    /// The default implementation returns `self` unchanged.
    #[cfg(std_io)]
    fn override_from_env(self) -> Self {
        self
    }

    /// Hook invoked exactly once, when the configuration singleton is first
    /// initialized — whether loaded from disk in [`RuntimeConfig::get`] or
    /// installed with [`RuntimeConfig::set`] / [`RuntimeConfig::try_set`].
    ///
    /// Use it to apply configuration to global state (stream policy, bundle
    /// installation, ...). Runs while the storage lock is held so that no
    /// concurrent [`RuntimeConfig::get`] can observe the configuration before
    /// the hook completed. Consequently the hook must not call
    /// [`RuntimeConfig::get`], [`RuntimeConfig::set`] or
    /// [`RuntimeConfig::try_set`] — that would deadlock.
    fn on_loaded(&self) {}

    /// Retrieves the current configuration, loading it from the current directory if not set.
    ///
    /// If no configuration is set, it attempts to load one from any of [`Config::file_names`] in
    /// the current directory or its parents. If no file is found, a default configuration is used.
    ///
    /// # Notes
    ///
    /// Calling this function is somewhat expensive, because of a global static lock. The config
    /// format is optimized for parsing, not for consumption. A good practice is to use a local
    /// static atomic value that you can populate with the appropriate value from the config
    /// during initialization.
    fn get() -> Arc<Self> {
        let mut state = Self::storage().lock();
        if state.as_ref().is_none() {
            cfg_if::cfg_if! {
                if #[cfg(std_io)] {
                    let config = Self::from_current_dir();
                    let config = config.override_from_env();
                } else {
                    let config = Self::default();
                }
            }

            let config = Arc::new(config);
            *state = Some(config.clone());
            // Still under the lock: a concurrent `get` must not observe the
            // configuration before the hook has run.
            config.on_loaded();

            return config;
        }

        state.as_ref().cloned().unwrap()
    }

    /// Sets the configuration to the provided value.
    ///
    /// # Panics
    /// Panics if the configuration has already been set or read, as it cannot be overridden.
    ///
    /// # Warning
    /// This method must be called at the start of the program, before any calls to
    /// [`Config::get`]. Attempting to set the configuration after it has been initialized will
    /// cause a panic.
    fn set(config: Self) {
        if !Self::try_set(config) {
            panic!("Cannot set the configuration multiple times.");
        }
    }

    /// Sets the configuration to the provided value, unless it has already been
    /// set or read — in which case the existing configuration is kept and
    /// `false` is returned.
    ///
    /// Use this from libraries that want to provide a computed default without
    /// overriding a configuration the application set first.
    fn try_set(config: Self) -> bool {
        let mut state = Self::storage().lock();
        if state.is_some() {
            return false;
        }
        let config = Arc::new(config);
        *state = Some(config.clone());
        // Still under the lock: see `get`.
        config.on_loaded();
        true
    }

    /// Save the default configuration to the provided file path.
    #[cfg(std_io)]
    fn save_default<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
        use std::io::Write;

        let config = Self::get();
        let content =
            toml::to_string_pretty(config.as_ref()).expect("Default config should be serializable");
        let mut file = std::fs::File::create(path)?;
        file.write_all(content.as_bytes())?;

        Ok(())
    }

    /// Loads configuration from any of [`Config::file_names`] in the current directory or its
    /// parents.
    ///
    /// Traverses up the directory tree until a valid configuration file is found or the root
    /// is reached. Returns a default configuration if no file is found.
    #[cfg(std_io)]
    fn from_current_dir() -> Self {
        // A deleted or unreadable cwd is not a reason to abort: there is simply
        // no configuration file to find from here.
        let Ok(mut dir) = std::env::current_dir() else {
            return Self::default();
        };

        loop {
            for name in Self::file_names() {
                if let Ok(content) = Self::from_file_path(dir.join(name)) {
                    return content;
                }
            }

            for (name, section) in Self::section_file_names() {
                if let Ok(content) = Self::from_section_file_path(dir.join(name), section) {
                    return content;
                }
            }

            if !dir.pop() {
                break;
            }
        }

        Self::default()
    }

    /// Loads configuration from a specified file path.
    ///
    /// A file that does not parse is reported and skipped rather than fatal:
    /// configuration keys change between releases, and a stale `cubecl.toml`
    /// left in a checkout must not abort the application that reads it.
    #[cfg(std_io)]
    fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;

        match toml::from_str(&content) {
            Ok(config) => Ok(config),
            Err(err) => {
                report_malformed(&alloc::format!(
                    "Ignoring {path:?}, which doesn't have the right format => {err}"
                ));
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData, err))
            }
        }
    }

    /// Loads configuration from a specific TOML section of the file at the given path.
    #[cfg(std_io)]
    fn from_section_file_path<P: AsRef<std::path::Path>>(
        path: P,
        section: &str,
    ) -> std::io::Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;

        let mut table: toml::Table = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => {
                report_malformed(&alloc::format!(
                    "Ignoring {path:?}, which doesn't have the right format => {err}"
                ));
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, err));
            }
        };

        let value = match table.remove(section) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    alloc::format!("Section '{section}' not found"),
                ));
            }
        };

        match value.try_into() {
            Ok(config) => Ok(config),
            Err(err) => {
                report_malformed(&alloc::format!(
                    "Ignoring section '{section}' of {path:?}, which doesn't have the right \
                     format => {err}"
                ));
                Err(std::io::Error::new(std::io::ErrorKind::InvalidData, err))
            }
        }
    }
}

#[cfg(all(test, std_io))]
mod tests {
    use super::*;

    #[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct Probe {
        #[serde(default)]
        cache: bool,
    }

    static PROBE: crate::sync::Mutex<Option<Arc<Probe>>> = crate::sync::Mutex::new(None);

    impl RuntimeConfig for Probe {
        fn storage() -> &'static crate::sync::Mutex<Option<Arc<Self>>> {
            &PROBE
        }
        fn file_names() -> &'static [&'static str] {
            &["probe.toml"]
        }
        fn section_file_names() -> &'static [(&'static str, &'static str)] {
            &[("host.toml", "probe")]
        }
    }

    /// A directory holding one config file, removed when the returned handle
    /// drops.
    fn scratch(file: &str, content: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(file), content).unwrap();
        dir
    }

    /// A field whose *type* changed is the failure mode that silently reverts a
    /// whole file to defaults: the file is found, so nothing looks wrong, but
    /// every setting in it is dropped. It has to surface as an error rather
    /// than a `None`-shaped miss.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_wrongly_typed_field_fails_the_whole_file() {
        let dir = scratch("probe.toml", "cache = \"target\"\n");

        let err = Probe::from_file_path(dir.path().join("probe.toml")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// Same, one level in: the section exists but does not deserialize.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_wrongly_typed_field_fails_the_whole_section() {
        let dir = scratch("host.toml", "[probe]\ncache = \"target\"\n");

        let err = Probe::from_section_file_path(dir.path().join("host.toml"), "probe").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// A host file that simply has no section for us is ordinary, not an error
    /// worth reporting — that is how a shared `burn.toml` looks to a crate it
    /// says nothing about.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_missing_section_is_quiet() {
        let dir = scratch("host.toml", "[other]\nvalue = 1\n");

        let err = Probe::from_section_file_path(dir.path().join("host.toml"), "probe").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    /// The good path still reads the value through.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn a_well_formed_section_parses() {
        let dir = scratch("host.toml", "[probe]\ncache = true\n");

        let config = Probe::from_section_file_path(dir.path().join("host.toml"), "probe").unwrap();
        assert!(config.cache);
    }
}
