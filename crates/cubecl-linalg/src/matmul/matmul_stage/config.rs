use crate::matmul::{config::MatmulConfig, matmul_tile::TmmConfig};

pub trait SmmConfig: MatmulConfig {
    type TmmConfig: TmmConfig;

    fn into_tmm_config(self) -> Self::TmmConfig;
}
