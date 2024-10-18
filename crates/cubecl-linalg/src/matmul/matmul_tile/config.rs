use crate::matmul::config::MatmulConfig;

pub trait TmmConfig: MatmulConfig {
    fn plane_dim(&self) -> u32;
}
