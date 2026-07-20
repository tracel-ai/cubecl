use alloc::string::{String, ToString};

/// Which named environment the process warms into.
///
/// An environment is one local store holding every cached namespace. Exactly
/// one is active at a time, so naming them lets a single checkout keep several
/// side by side.
///
/// ```toml
/// [environment]
/// name = "h100"
/// ```
///
/// `CUBECL_ENVIRONMENT` overrides it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentConfig {
    /// Name of the environment to activate when the configuration loads.
    #[serde(default = "default_name")]
    pub name: String,
}

fn default_name() -> String {
    cubecl_environment::environment::DEFAULT.to_string()
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            name: default_name(),
        }
    }
}
