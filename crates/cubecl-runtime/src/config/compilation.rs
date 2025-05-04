use super::logger::{BinaryLogLevel, LoggerConfig};

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    #[serde(default)]
    pub logger: LoggerConfig<CompilationLogLevel>,
}

pub type CompilationLogLevel = BinaryLogLevel;
