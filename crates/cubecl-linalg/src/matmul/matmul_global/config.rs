use crate::matmul::config::MatmulConfig;
use crate::matmul::matmul_stage::SmmConfig;

pub trait GmmConfig: MatmulConfig {
    type SmmConfig: SmmConfig;

    fn into_smm_config(self) -> Self::SmmConfig;
}
