use crate::matmul::config::ComptimeConfig;
use crate::matmul::matmul_tile::TmmConfig;

pub trait SmmConfig: ComptimeConfig {
    type TmmConfig: TmmConfig;

    fn into_tmm_config(self) -> Self::TmmConfig;
}
