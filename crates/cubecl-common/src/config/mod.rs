/// Reusable logger configuration and sink management.
pub mod logger;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

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
    fn storage() -> &'static spin::Mutex<Option<Arc<Self>>>;

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

            *state = Some(Arc::new(config));
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
        let mut state = Self::storage().lock();
        if state.is_some() {
            panic!("Cannot set the configuration multiple times.");
        }
        *state = Some(Arc::new(config));
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
        let mut dir = std::env::current_dir().unwrap();

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
    #[cfg(std_io)]
    fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err}"),
        };

        Ok(config)
    }

    /// Loads configuration from a specific TOML section of the file at the given path.
    #[cfg(std_io)]
    fn from_section_file_path<P: AsRef<std::path::Path>>(
        path: P,
        section: &str,
    ) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut table: toml::Table = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err}"),
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

        let config: Self = match value.try_into() {
            Ok(val) => val,
            Err(err) => {
                panic!("The section '{section}' doesn't have the right format => {err}")
            }
        };

        Ok(config)
    }
}
