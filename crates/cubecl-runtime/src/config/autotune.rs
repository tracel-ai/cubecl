use super::logger::{LogLevel, LoggerConfig};

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AutotuneConfig {
    #[serde(default)]
    pub logger: LoggerConfig<AutotuneLogLevel>,
    #[serde(default)]
    pub level: AutotuneLevel,
    #[serde(default)]
    pub cache: AutotuneCache,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneCache {
    Local,
    #[default]
    Target,
    Global,
    File(std::path::PathBuf),
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLogLevel {
    #[serde(rename = "disabled")]
    Disabled,
    #[default]
    #[serde(rename = "minimal")]
    Minmal,
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for AutotuneLogLevel {}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLevel {
    #[serde(rename = "minimal")]
    Minimal,
    #[default]
    #[serde(rename = "balanced")]
    Medium,
    #[serde(rename = "more")]
    More,
    #[serde(rename = "full")]
    Full,
}

impl AutotuneCache {
    pub fn root(&self) -> std::path::PathBuf {
        match self {
            AutotuneCache::Local => std::env::current_dir().unwrap(),
            AutotuneCache::Target => {
                let dir_original = std::env::current_dir().unwrap();
                let mut dir = dir_original.clone();

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
            AutotuneCache::Global => dirs::config_local_dir().unwrap(),
            AutotuneCache::File(path_buf) => path_buf.clone(),
        }
    }
}
