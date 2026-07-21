use etcetera::BaseStrategy;

/// Cache location options.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CacheConfig {
    /// Stores cache in the current working directory.
    #[serde(rename = "local")]
    Local,

    /// Stores cache in the project's `target/environment` directory
    /// (default).
    #[default]
    #[serde(rename = "target")]
    Target,

    /// Stores cache in the system's user cache directory.
    #[serde(rename = "global")]
    Global,

    /// Stores cache under a user-specified directory. The environment
    /// database file is placed inside it, not at this path.
    #[serde(rename = "directory")]
    Directory(std::path::PathBuf),
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
                let start = match std::env::current_dir() {
                    Ok(dir) => dir,
                    // Same condition `Local` reports: a daemon or test harness
                    // whose cwd was deleted has no current directory to search
                    // from.
                    Err(err) => {
                        log::warn!(
                            "cubecl cache: no current directory ({err}); using the user cache"
                        );
                        return user_cache_dir();
                    }
                };

                // The *outermost* Cargo.toml, not the first one found walking
                // up: a workspace member has its own manifest, but `target/`
                // lives at the workspace root, so stopping at the innermost
                // manifest would scatter caches into `member/target`.
                let mut root = None;
                let mut dir = start.as_path();
                loop {
                    if let Ok(true) = std::fs::exists(dir.join("Cargo.toml")) {
                        root = Some(dir);
                    }
                    match dir.parent() {
                        Some(parent) => dir = parent,
                        None => break,
                    }
                }

                match root {
                    Some(root) => root.join("target").join("environment"),
                    // No Cargo.toml anywhere above cwd — this is a bundled or
                    // installed application (Tauri, GUI app, CLI installed via
                    // cargo install, etc.) running outside a workspace. Joining
                    // "target" onto the original cwd became `/target` when cwd
                    // was `/`, which fails on most platforms with EROFS and
                    // cascaded a directory failure into the whole autotune
                    // pipeline. Use the platform-appropriate user cache
                    // directory instead.
                    None => user_cache_dir(),
                }
            }
            // The cache directory, not the configuration directory: this is
            // regenerable data, and XDG says so. It also matches the `Target`
            // fallback right above, which already used `cache_dir`.
            Self::Global => user_cache_dir(),
            Self::Directory(path_buf) => path_buf.clone(),
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
