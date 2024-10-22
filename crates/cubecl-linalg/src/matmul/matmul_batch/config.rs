use crate::matmul::config::{ComptimeConfig, MatmulConfig};
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::Ident;
use crate::matmul::stage_dim::StageDim;

pub trait BmmConfig: ComptimeConfig + MatmulConfig {
    type GmmConfig: GmmConfig;

    fn to_gmm_config(&self) -> Self::GmmConfig;
    fn stage_dim(&self, ident: Ident) -> StageDim;

    fn cube_count_x(&self) -> u32;
    fn cube_count_y(&self) -> u32;

    fn max_m(&self) -> u32;
    fn max_n(&self) -> u32;
    fn max_batches(&self) -> u32;
}
