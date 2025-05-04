use alloc::sync::Arc;

use super::{autotune::AutotuneConfig, compilation::CompilationConfig, profiling::ProfilingConfig};

static CUBE_GLOBAL_CONFIG: spin::Mutex<Option<Arc<GlobalConfig>>> = spin::Mutex::new(None);

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GlobalConfig {
    #[serde(default)]
    pub profiling: ProfilingConfig,
    #[serde(default)]
    pub autotune: AutotuneConfig,
    #[serde(default)]
    pub compilation: CompilationConfig,
}

impl GlobalConfig {
    pub fn get() -> Arc<Self> {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        if let None = state.as_ref() {
            let config = Self::from_current_dir();
            *state = Some(Arc::new(config));
        }

        state.as_ref().cloned().unwrap()
    }

    pub fn set(config: Self) {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        *state = Some(Arc::new(config));
    }

    fn from_current_dir() -> Self {
        let mut dir = std::env::current_dir().unwrap();

        loop {
            if let Ok(content) = Self::from_file_path(dir.join("cubecl.toml")) {
                return content;
            }

            if let Ok(content) = Self::from_file_path(dir.join("CubeCL.toml")) {
                return content;
            }

            if !dir.pop() {
                break;
            }
        }

        Self::default()
    }

    fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err:?}"),
        };

        Ok(config)
    }
}
