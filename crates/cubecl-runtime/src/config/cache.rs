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
                let dir_original =
                    std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
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

                // No Cargo.toml anywhere above cwd — this is a bundled or
                // installed application (Tauri, GUI app, CLI installed via
                // cargo install, etc.) running outside a workspace. The
                // previous fallback of `dir_original.join("target")` became
                // `/target` when cwd was `/`, which fails on most platforms
                // with EROFS (read-only system volume on macOS, root-owned
                // on Linux) and cascaded a `CacheFile::new` directory
                // failure into the whole autotune pipeline. Use the
                // platform-appropriate user cache directory instead.
                if let Some(cache) = dirs::cache_dir() {
                    return cache.join("cubecl");
                }
                dir_original.join("target")
            }
            Self::Global => dirs::config_local_dir().unwrap(),
            Self::File(path_buf) => path_buf.clone(),
        }
    }
}
