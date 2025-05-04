/// Cache location options.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CacheConfig {
    /// Stores cache in the current working directory.
    #[serde(rename = "local")]
    Local,

    /// Stores cache in the project's `target` directory (default).
    #[default]
    #[serde(rename = "target")]
    Target,

    /// Stores cache in the system's local configuration directory.
    #[serde(rename = "global")]
    Global,

    /// Stores cache in a user-specified file path.
    #[serde(rename = "file")]
    File(std::path::PathBuf),
}

impl CacheConfig {
    /// Returns the root directory for the cache.
    pub fn root(&self) -> std::path::PathBuf {
        match self {
            Self::Local => std::env::current_dir().unwrap(),
            Self::Target => {
                let dir_original = std::env::current_dir().unwrap();
                let mut dir = dir_original.clone();

                // Search for Cargo.toml in parent directories to locate project root.
                loop {
                    if let Ok(true) = std::fs::exists(dir.join("Cargo.toml")) {
                        return dir.join("target");
                    }

                    if !dir.pop() {
                        break;
                    }
                }

                dir_original.join("target")
            }
            Self::Global => dirs::config_local_dir().unwrap(),
            Self::File(path_buf) => path_buf.clone(),
        }
    }
}
