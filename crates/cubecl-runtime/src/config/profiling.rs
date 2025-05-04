use super::logger::{LogLevel, LoggerConfig};

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProfilingConfig {
    #[serde(default)]
    pub logger: LoggerConfig<ProfilingLogLevel>,
}

impl LogLevel for ProfilingLogLevel {}

#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProfilingLogLevel {
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "basic")]
    Basic,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "full")]
    Full,
}
