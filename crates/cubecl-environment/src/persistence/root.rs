use etcetera::BaseStrategy;

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

    /// Stores cache in the system's user cache directory.
    #[serde(rename = "global")]
    Global,

    /// Stores cache in a user-specified file path.
    #[serde(rename = "file")]
    File(std::path::PathBuf),
}

impl CacheConfig {
    /// Returns the root directory for the cache.
    ///
    /// Every arm degrades rather than fails: none of these look-ups is under
    /// the application's control, and a cache root that can't be resolved must
    /// cost a recompute, not abort the process. A root that turns out to be
    /// unwritable is handled one level down, in
    /// [`Database::open_at`](crate::persistence::Database::open_at).
    pub fn root(&self) -> std::path::PathBuf {
        match self {
            // A daemon or test harness whose cwd was deleted has no current
            // directory at all.
            Self::Local => std::env::current_dir().unwrap_or_else(|err| {
                log::warn!("cubecl cache: no current directory ({err}); using the user cache");
                user_cache_dir()
            }),
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
                if let Ok(strategy) = etcetera::choose_base_strategy() {
                    return strategy.cache_dir().join("cubecl");
                }
                dir_original.join("target")
            }
            // The cache directory, not the configuration directory: this is
            // regenerable data, and XDG says so. It also matches the `Target`
            // fallback right above, which already used `cache_dir`.
            Self::Global => user_cache_dir(),
            Self::File(path_buf) => path_buf.clone(),
        }
    }
}

/// The user cache directory, or a temporary one when the platform can't name
/// it — a systemd unit with no `HOME`, a distroless container.
fn user_cache_dir() -> std::path::PathBuf {
    match etcetera::choose_base_strategy() {
        Ok(strategy) => strategy.cache_dir().join("cubecl"),
        Err(err) => {
            log::warn!(
                "cubecl cache: no user cache directory ({err}); \
                 falling back to the temporary directory"
            );
            std::env::temp_dir().join("cubecl")
        }
    }
}
