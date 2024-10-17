use crate::matmul::config::ComptimeConfig;
use crate::matmul::matmul_stage::SmmConfig;

pub trait GmmConfig: ComptimeConfig {
    type SmmConfig: SmmConfig;

    fn to_smm_config(&self) -> Self::SmmConfig;
}

pub trait ViewConfig: ComptimeConfig {}
