use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::environment;
use cubecl_runtime::config::CubeClRuntimeConfig;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// How a command line names an environment, resolved the way the runtime
/// resolves its own: `[cubecl.environment]` in the `cubecl.toml` or
/// `burn.toml` found from the working directory decides where environments
/// live, so run inside a project, a name reads what that project's runs
/// write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvironmentName {
    /// The one the runtime runs in.
    Active,
    /// A file, named by its path.
    Path(PathBuf),
    /// A named environment: `<root>/<name>.db`.
    Named(String),
}

/// A directory of environment files, and the extension that marks one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvironmentDirectory {
    pub path: PathBuf,
    /// With or without its dot.
    pub extension: String,
}

impl EnvironmentName {
    /// How [`Active`](Self::Active) is spelled.
    pub const ACTIVE: &str = "active";

    /// The file this names, whether or not it exists yet.
    pub fn resolve(&self) -> PathBuf {
        load_configuration();
        match self {
            Self::Active => environment::path(),
            Self::Path(path) => path.clone(),
            Self::Named(name) => environment::root().join(environment::file_name(name)),
        }
    }
}

/// `active`, then an existing path, then a name: a name is never also a file
/// in the working directory worth reaching for. Every text names something,
/// so reading one cannot fail.
impl FromStr for EnvironmentName {
    type Err = Infallible;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        let path = PathBuf::from(name);
        Ok(if name == Self::ACTIVE {
            Self::Active
        } else if path.exists() {
            Self::Path(path)
        } else {
            Self::Named(name.to_string())
        })
    }
}

impl EnvironmentDirectory {
    /// Where the runtime keeps its named environments.
    pub fn runtime() -> Self {
        load_configuration();
        Self {
            path: environment::root(),
            extension: environment::EXTENSION.to_string(),
        }
    }

    /// The files its extension marks, read with or without its dot; none
    /// when the directory does not exist — a machine that never built has no
    /// environments.
    pub(crate) fn files(&self) -> std::io::Result<Vec<PathBuf>> {
        let entries = match std::fs::read_dir(&self.path) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let extension = self.extension.trim_start_matches('.');
        Ok(entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(extension))
            .collect())
    }
}

/// An environment's name: its file's stem, what `--env` takes to open it
/// again.
pub(crate) fn stem(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

/// Load the runtime's configuration, which is what points the environment at
/// its root and name. An application that installed its own before this is
/// read keeps it.
fn load_configuration() {
    CubeClRuntimeConfig::get();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_active_then_a_path_then_a_name() {
        let dir = std::env::temp_dir();
        let path = dir.to_string_lossy();
        let name = |text: &str| text.parse::<EnvironmentName>();
        assert_eq!(name("active"), Ok(EnvironmentName::Active));
        assert_eq!(name(&path), Ok(EnvironmentName::Path(dir.clone())));
        assert_eq!(
            name("building"),
            Ok(EnvironmentName::Named("building".to_string()))
        );
    }
}
